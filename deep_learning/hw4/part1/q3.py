import copy
import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from torch.utils import data

from getimagenetclasses import parseclasslabel, parsesynsetwords


def get_label(path_words_txt, path_xml):
    indicestosynsets, synsetstoindices, synsetstoclassdescr = parsesynsetwords(
        path_words_txt
    )
    idx, label_id = parseclasslabel(path_xml, synsetstoindices)
    name = synsetstoclassdescr[indicestosynsets[idx]]
    return dict(label_idx=idx, label_name=name, label_id=label_id)


def show_data(data, nrows=8, ncols=8, size=3):
    data = copy.deepcopy(data)
    np.random.seed(42)
    idxs = np.random.choice(len(data), size=nrows * ncols, replace=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, ncols * size))
    for i, j in enumerate(idxs):
        d = data[j]
        image = plt.imread(d.pop("path_image"))
        axes[i // ncols, i % ncols].imshow(image)
        axes[i // ncols, i % ncols].text(0, 0, str(d), bbox=dict(facecolor="white"))
    plt.show()


class PickleSaver:
    def __init__(self, path):
        self.path = path

    def save(self, obj):
        with open(self.path, "wb") as f:
            return pickle.dump(obj, f)

    def load(self):
        with open(self.path, "rb") as f:
            return pickle.load(f)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data(image_dir, label_dir, path_words_txt, save_path):
    saver = PickleSaver(save_path)
    if not saver.path.exists():
        data = []
        for path_image in tqdm.tqdm(list(image_dir.iterdir())):
            assert path_image.suffix == ".JPEG"
            name = path_image.stem
            path_label = (label_dir / name).with_suffix(".xml")
            assert path_label.exists()
            d = get_label(path_words_txt, path_label)
            d.update(dict(path_image=path_image))
            data.append(d)
        saver.save(data)
    return saver.load()


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open(self.data[item]["path_image"])
        image = image.convert("RGB")
        if self.transform is not None:
            image: torch.Tensor = self.transform(image)

        # image = torch.einsum("ijk->jki", image)
        return image, self.data[item]["label_idx"]

    def __len__(self):
        return len(self.data)


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
        for inputs, labels in tqdm.tqdm(loader):
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


def random_sample(lst, size):
    np.random.seed(42)
    idxs = np.random.choice(len(lst), size=size, replace=False)
    return [lst[i] for i in idxs]


def main(arch, data_dir=".", data_limit=None, do_normalize=True):
    data_dir = pathlib.Path(data_dir)
    image_dir = data_dir / "imagespart"
    label_dir = data_dir / "val"
    path_words_txt = data_dir / "synset_words.txt"
    image_size = 330
    batch_size = 32
    criterion = torch.nn.CrossEntropyLoss()

    save_path = data_dir / "data.pkl"
    data = get_data(image_dir, label_dir, path_words_txt, save_path)
    if data_limit is not None:
        data = random_sample(data, size=data_limit)
    # show_data(data)

    transforms = [
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
    ]
    if do_normalize:
        transforms.append(
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        )
    ds = MyDataset(data, torchvision.transforms.Compose(transforms))
    loader = torch.utils.data.DataLoader(ds, batch_size, shuffle=False)
    x, y = next(iter(loader))
    print(x.shape, y.shape)  # torch.Size([32, 3, 224, 224]) torch.Size([32])

    model = arch(pretrained=True)
    model.to(get_device())
    # print(model)

    result = run_epoch(model, loader, criterion, optimizer=None, training=False)
    print(result)


if __name__ == "__main__":
    main(arch=torchvision.models.resnet18, data_limit=250)  # {'loss': 1.47, 'acc': 0.7}
    main(
        arch=torchvision.models.resnet34, data_limit=250
    )  # {'loss': 1.255, 'acc': 0.728}
