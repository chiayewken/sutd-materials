import shutil
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import make_grid
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, num_labels=10, embed_dim=10):
        super().__init__()

        # use an embedding layer for the layer
        self.label_emb = nn.Embedding(num_labels, embed_dim)

        self.model = nn.Sequential(
            nn.Linear(784 + embed_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        # concatenates x and c, so the condition is given as input to the model appended to the image x
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Tanh(),
        )
        self.conv = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=1
        )

    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat(
            [z, c], 1
        )  # concatenates z and c, so the condition is given as input to the model appended to z
        out = self.model(x)

        out = out.view(x.size(0), 1, 28, 28)
        out = self.conv(out)
        out = torch.tanh(out)

        return out.view(x.size(0), 28, 28)


def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(
        torch.from_numpy(np.random.randint(0, 10, batch_size)).long()
    ).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


def discriminator_train_step(
    batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels
):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())

    # train with fake images
    # generate random noise image
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(
        torch.from_numpy(np.random.randint(0, 10, batch_size)).long()
    ).cuda()
    # feed the noise to the generator and get the output
    fake_images = generator(z, fake_labels)
    # evaluate this output
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())

    # optimize the sum of both losses
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


class SmallImageDataset(Dataset):
    def __init__(self, train: bool, transform=None):
        self.train = train
        self.transform = transform
        self.root = Path("/tmp/hymenoptera_data")
        self.folder = self.download()
        self.class_names = self.folder.classes

    def download(self) -> ImageFolder:
        url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
        if not self.root.exists():
            download_and_extract_archive(
                url, download_root="/tmp", extract_root=str(self.root.parent)
            )
        data_split = "train" if self.train else "val"
        folder = ImageFolder(str(self.root / data_split), transform=self.transform)
        return folder

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        image, label = self.folder[i]
        assert isinstance(image, torch.Tensor)
        c, h, w = image.shape
        assert h == w
        return image, label

    def __len__(self) -> int:
        return len(self.folder)

    def plot_samples(self, path="sample.png", num_samples=10):
        np.random.seed(42)
        indices = np.random.choice(len(self), size=num_samples, replace=False)
        images = []
        labels = []
        for i in indices:
            x, y = self[i]
            images.append(x)
            labels.append(y)

        stacked = torch.stack(images)
        info = dict(
            min=stacked.min(),
            max=stacked.max(),
            mean=stacked.mean(),
            std=stacked.std(),
            # labels=[self.class_names[i] for i in labels],
            labels=[self.class_names[int(np.argmax(i))] for i in labels],
        )
        print(info)
        save_image(images, path, normalize=True)


def get_transform(image_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            # transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ]
    )
    # # we load it much like in the previous CNN tutorial and standardize the values
    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)]
    # )
    return transform


def get_dataset(image_size: int, name="fashion"):
    transform = get_transform(image_size)
    if name == "fashion":
        dataset = FashionMNIST(
            root="./data", train=True, transform=transform, download=True
        )
    else:
        dataset = SmallImageDataset(train=True, transform=transform)

    x, y = dataset[0]  # [1, 28, 28], int
    assert isinstance(x, torch.Tensor)
    assert x.min() >= -1.0
    assert x.max() <= 1.0
    assert list(x.shape) == [1, image_size, image_size]
    assert type(y) == int
    return dataset


def main():
    num_epochs = 30
    image_size = 28

    folder_samples = Path("samples")
    if folder_samples.exists():
        shutil.rmtree(folder_samples)
    folder_samples.mkdir()

    dataset = get_dataset(image_size, name="small")
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)

    # for each epoch
    g_loss, d_loss = None, None
    for epoch in range(num_epochs):
        print("Starting epoch {}...".format(epoch))
        for i, (images, labels) in enumerate(data_loader):
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            # train the generator
            generator.train()
            batch_size = real_images.size(0)
            # train the discriminator
            d_loss = discriminator_train_step(
                len(real_images),
                discriminator,
                generator,
                d_optimizer,
                criterion,
                real_images,
                labels,
            )

            g_loss = generator_train_step(
                batch_size, discriminator, generator, g_optimizer, criterion
            )

        generator.eval()
        print("g_loss: {}, d_loss: {}".format(g_loss, d_loss))
        # generate random noise to feed to the generator
        z = Variable(torch.randn(9, 100)).cuda()

        # get all the labels
        labels = Variable(torch.from_numpy(np.arange(9))).long().cuda()

        # generate a new mage for each of the labels based on the label and the noise z
        sample_images = generator(z, labels).unsqueeze(1).data.cpu()

        # display the images
        grid = make_grid(sample_images, nrow=3, normalize=True).permute(1, 2, 0).numpy()
        plt.imshow(grid)
        path = (folder_samples / str(epoch).zfill(3)).with_suffix(".jpg")
        path.parent.mkdir(exist_ok=True)
        plt.savefig(str(path))
        plt.show()


if __name__ == "__main__":
    main()
