import copy
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
import tqdm
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path_anno_txt, images_dir: pathlib.Path, transform=None):
        print("Dataset reading:", path_anno_txt, images_dir)
        self.data = []
        with open(path_anno_txt) as f:
            for line in f:
                name_image, label = line.strip().split()
                path_image = images_dir / name_image
                assert path_image.exists()
                label = int(label)
                self.data.append((path_image, label))
        labels = [l for p, l in self.data]
        self.num_unique_labels = len(set(labels))
        assert max(labels) + 1 == self.num_unique_labels
        assert min(labels) == 0
        print("Num data:", len(self.data))
        self.transform = transform

    def __getitem__(self, item):
        path_image, label = self.data[item]
        image: Image.Image = Image.open(path_image)
        image = image.convert("RGB")
        if self.transform is not None:
            image: torch.Tensor = self.transform(image)

        # image = torch.einsum("ijk->jki", image)
        return image, label

    def __len__(self):
        return len(self.data)

    def show_data(self, nrows=8, ncols=8, size=3):
        np.random.seed(42)
        idxs = np.random.choice(len(self), size=nrows * ncols, replace=False)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, ncols * size))
        for i, j in enumerate(idxs):
            path_image, label = self.data[j]
            image = plt.imread(path_image)
            axes[i // ncols, i % ncols].imshow(image)
            axes[i // ncols, i % ncols].text(
                0, 0, str(label), bbox=dict(facecolor="white")
            )
        plt.show()

    def get_loader(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def get_model(model_fn, num_labels, pretrained, train_last_2_layers_only):
    model = model_fn(pretrained=pretrained)
    print(model)
    fc = model.fc
    num_features = fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_features, num_labels),
        torch.nn.ReLU(),
        torch.nn.Linear(num_labels, num_labels),
    )
    print(model)

    if train_last_2_layers_only:
        for param in model.parameters():
            param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    params_train = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Param requires grad:", name)
            params_train.append(param)

    return model, params_train


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def run_epoch(net, loader, criterion, optimizer, training):
    device = get_device()
    if training:
        net.train()
        context_grad = torch.enable_grad
    else:
        net.eval()
        context_grad = torch.no_grad

    correct = 0
    total = 0
    losses = []
    with context_grad():
        for inputs, labels in loader:
            if training:
                optimizer.zero_grad()
            inputs = inputs.to(device)  # torch.float32
            labels = labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            losses.append(loss.item())
            if training:
                loss.backward()
                optimizer.step()

    acc = np.round(correct / total, decimals=3)
    loss = np.round(np.mean(losses), decimals=3)
    return dict(loss=loss, acc=acc)


def train(net, criterion, optimizer, num_epochs, loaders, earlystop):
    best_loss = 1e9
    results = dict(best_weights=None, metrics=[])

    for e in tqdm.tqdm(range(num_epochs)):
        metrics = dict(
            train=run_epoch(net, loaders["train"], criterion, optimizer, training=True),
            val=run_epoch(net, loaders["val"], criterion, optimizer, training=False),
            test=run_epoch(net, loaders["test"], criterion, optimizer, training=False),
        )
        results["metrics"].append(metrics)
        print(dict(epoch=e, metrics=metrics))

        loss_monitor = metrics["val"]["loss"]
        if loss_monitor < best_loss:
            best_loss = loss_monitor
            results["best_weights"] = copy.deepcopy(net.state_dict())
        elif earlystop:
            break

    return results


class PickleSaver:
    def __init__(self, path):
        self.path = path

    def save(self, obj):
        with open(self.path, "wb") as f:
            return pickle.dump(obj, f)

    def load(self):
        with open(self.path, "rb") as f:
            return pickle.load(f)


def main(data_dir, pretrained, train_last_2_layers_only, save_path="results.pkl"):
    data_dir = pathlib.Path(data_dir)
    images_dir = data_dir / "flowers_data/jpg"
    image_size = 224
    batch_size = 32
    num_epochs = 10
    crit = torch.nn.CrossEntropyLoss()
    model_fn = torchvision.models.resnet18

    print("Num images:", len(list(images_dir.iterdir())))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    datasets = {}
    for name in ["test", "train", "val", "valtest"]:
        datasets[name] = MyDataset(data_dir / f"{name}file.txt", images_dir, transform)
    print("Datasets:", datasets.keys())
    # datasets["train"].show_data()

    loaders = {name: ds.get_loader(batch_size) for name, ds in datasets.items()}
    x, y = next(iter(loaders["train"]))
    print(x.shape, y.shape)

    model, params_train = get_model(
        model_fn,
        num_labels=datasets["train"].num_unique_labels,
        pretrained=pretrained,
        train_last_2_layers_only=train_last_2_layers_only,
    )
    model.to(get_device())
    opt = torch.optim.SGD(params_train, lr=1e-3, momentum=0.9)
    print("Test:", run_epoch(model, loaders["test"], crit, opt, training=False))
    results = train(model, crit, opt, num_epochs, loaders, earlystop=True)
    model.load_state_dict(results.pop("best_weights"))
    print("Test:", run_epoch(model, loaders["test"], crit, opt, training=False))
    saver = PickleSaver(save_path)
    saver.save(results)


if __name__ == "__main__":
    main(
        data_dir=".",
        pretrained=False,
        train_last_2_layers_only=False,
        save_path="results1.pkl",
    )  # Test: {'loss': 2.901, 'acc': 0.251}
    main(
        data_dir=".",
        pretrained=True,
        train_last_2_layers_only=False,
        save_path="results2.pkl",
    )  # Test: {'loss': 0.522, 'acc': 0.881}
    main(
        data_dir=".",
        pretrained=True,
        train_last_2_layers_only=True,
        save_path="results3.pkl",
    )  # Test: {'loss': 1.418, 'acc': 0.679}
