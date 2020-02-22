import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from sklearn import model_selection
from tqdm import tqdm


def get_dataset(data_dir, is_train):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )
    ds = torchvision.datasets.FashionMNIST(
        data_dir, train=is_train, download=True, transform=transform
    )
    print("Dataset:", len(ds))
    return ds


def show_data(ds, nrows=8, ncols=8, size=3):
    np.random.seed(42)
    idxs = np.random.choice(len(ds), size=nrows * ncols, replace=False)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, ncols * size))
    for i, j in enumerate(idxs):
        x, y = ds[j]
        x = torch.einsum("ijk->jk", x)
        x = x.numpy()
        axes[i // ncols, i % ncols].imshow(x)
        axes[i // ncols, i % ncols].text(0, 0, str(y), bbox=dict(facecolor="white"))
    plt.show()


def get_trainval_samplers(ds, val_size):
    idxs_trn, idxs_val = model_selection.train_test_split(
        np.arange(len(ds)), test_size=val_size, random_state=42
    )
    sampler_trn = torch.utils.data.SubsetRandomSampler(idxs_trn)
    sampler_val = torch.utils.data.SubsetRandomSampler(idxs_val)
    return sampler_trn, sampler_val


class Net(torch.nn.Module):
    def __init__(self, dims_in, dims_out, dims_fc1=300, dims_fc2=100):
        super().__init__()
        self.dims_in = dims_in
        self.fc1 = torch.nn.Linear(dims_in, dims_fc1)
        self.fc2 = torch.nn.Linear(dims_fc1, dims_fc2)
        self.fc3 = torch.nn.Linear(dims_fc2, dims_out)

    def forward(self, x):
        x = x.view(-1, self.dims_in)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x


def get_data_dims(ds):
    x, y = ds[0]
    dims_in = x.numel()
    unique_labels = set()
    for i in range(len(ds)):
        x, y = ds[i]
        unique_labels.add(y)
    dims_out = len(unique_labels)
    print("Dims in/out:", dims_in, dims_out)
    return dims_in, dims_out


def evaluate(net, loader, criterion, device):
    net.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            losses.append(criterion(outputs, labels).cpu())
    acc = np.round(correct / total, decimals=3)
    loss = np.round(np.mean(losses), decimals=3)
    return dict(loss=loss, acc=acc)


def train(net, criterion, optimizer, num_epochs, loaders, lr, device, earlystop):
    best_loss = 1e9
    results = dict(best_weights=None, metrics=[], learn_rate=lr)
    for e in tqdm(range(num_epochs)):
        net.train()
        for data in loaders["trn"]:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        metrics = {}
        for _split in loaders.keys():
            metrics[_split] = evaluate(net, loaders[_split], criterion, device)
        results["metrics"].append(metrics)
        # metrics = dict(trn=dict(loss, acc), val=dict(loss,acc), test=dict(loss,acc))

        print(dict(epoch=e, metrics=metrics))
        # INCORRECTLY, here use validation set equal to the test set
        loss_monitor = metrics["test"]["loss"]
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


def main(data_dir, batch_size, epochs, lr, earlystop, save_path="results.pkl"):
    ds_trn = get_dataset(data_dir, is_train=True)
    ds_test = get_dataset(data_dir, is_train=False)
    show_data(ds_trn)

    sampler_trn, sampler_val = get_trainval_samplers(ds_trn, val_size=0.1)
    loaders = dict(
        trn=torch.utils.data.DataLoader(ds_trn, batch_size, sampler=sampler_trn),
        val=torch.utils.data.DataLoader(ds_trn, batch_size, sampler=sampler_val),
        test=torch.utils.data.DataLoader(ds_test, batch_size),
    )

    dims_in, dims_out = get_data_dims(ds_trn)
    net = Net(dims_in, dims_out)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    print("Device:", device)
    results = train(net, criterion, optimizer, epochs, loaders, lr, device, earlystop)
    net.load_state_dict(results.pop("best_weights"))
    print("Best test results:", evaluate(net, loaders["test"], criterion, device))
    saver = PickleSaver(save_path)
    saver.save(results)

    # Best test results: {'loss': 0.418, 'acc': 0.852}

    """
    • an answer for the following: When you train a deep neural net, then you
    get after every epoch one model (actually after every minibatch). Why
    you should not select the best model over all epochs on the test dataset?
    """


if __name__ == "__main__":
    main(data_dir="data", batch_size=32, epochs=100, lr=1e-3, earlystop=True)
