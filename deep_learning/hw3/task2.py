import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision
from sklearn import model_selection
from tqdm import tqdm


def get_ds(data_dir, is_train):
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


def get_trainval_samplers(ds, test_size):
    idxs_trn, idxs_val = model_selection.train_test_split(
        np.arange(len(ds)), test_size=0.1, random_state=42
    )
    sampler_trn = torch.utils.data.SubsetRandomSampler(idxs_trn)
    sampler_val = torch.utils.data.SubsetRandomSampler(idxs_val)
    return sampler_trn, sampler_val


class Net(torch.nn.Module):
    def __init__(self, dims_in, dims_out, dims_hidden=100):
        super().__init__()
        self.dims_in = dims_in
        self.fc1 = torch.nn.Linear(dims_in, dims_hidden)
        self.fc2 = torch.nn.Linear(dims_hidden, dims_hidden)
        self.fc3 = torch.nn.Linear(dims_hidden, dims_out)

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
    for x, y in ds_trn:
        unique_labels.add(y)
    dims_out = len(unique_labels)
    print("Dims in/out:", dims_in, dims_out)
    return dims_in, dims_out


def train(net, criterion, optimizer, num_epochs):
    for e in tqdm(range(num_epochs)):
        losses = []
        for i, data in enumerate(loader_trn):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("Epoch:", e, "Loss:", np.mean(losses))


if __name__ == "__main__":
    data_dir = "data"
    batch_size = 8
    num_epochs = 10

    ds_trn = get_ds(data_dir, is_train=True)
    ds_test = get_ds(data_dir, is_train=False)
    show_data(ds_trn)

    sampler_trn, sampler_val = get_trainval_samplers(ds_trn, test_size=0.1)
    loader_trn = torch.utils.data.DataLoader(ds_trn, batch_size, sampler=sampler_trn)
    loader_val = torch.utils.data.DataLoader(ds_trn, batch_size, sampler=sampler_val)
    loader_test = torch.utils.data.DataLoader(ds_test, batch_size)

    dims_in, dims_out = get_data_dims(ds_trn)
    net = Net(dims_in, dims_out)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    train(net, criterion, optimizer, num_epochs)
