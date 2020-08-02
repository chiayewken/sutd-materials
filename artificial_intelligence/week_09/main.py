"""
Question 1 -- Generate images based on a condition
Task: develop a generative model (either cGAN or cVaE)
that can generate images based on a class label (bee or ant).

a) Training dataset: small subset of ImageNet:
https://download.pytorch.org/tutorial/hymenoptera_data.zip.
You can leverage the ImageFolder class as demonstrated here:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html.
Since we don't need a test set for generation,
you can combine both test and training datasets.

To handle using custom datasets, torchvision provides a datasets.ImageFolder class.
ImageFolder expects data to be stored in the following way:

root/class_x/xxy.png
root/class_x/xxz.jpg
root/class_y/123.jpeg
root/class_y/nsdf3.png
root/class_y/asd932_.jpg

b) Normalize the training data and perform data augmentation.

For those interested in knowing how to calculate the means for normalizing,
please refer to this excellent resource here:
https://github.com/bentrevett/pytorch-image-classification/blob/master/5%20-%20ResNet.ipynb
and the previously mentioned link.
"""
import shutil

from pydantic import BaseModel
from pathlib import Path
import numpy as np
from typing import Tuple, List

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import save_image


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

    def convert_one_hot(self, i: int) -> torch.Tensor:
        one_hot = torch.zeros(len(self.class_names))
        one_hot[i] = 1.0
        return one_hot

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, i_label = self.folder[i]
        label = self.convert_one_hot(i_label)
        assert isinstance(image, torch.Tensor)
        c, h, w = image.shape
        assert h == w
        assert label.shape == (len(self.class_names),)
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


class Config(BaseModel):
    model_name: str = "cvae"
    in_channels: int = 3
    num_classes: int = 2
    latent_dim: int = 128
    hidden_dims: List[int] = [32, 64, 128]
    image_size: int = 64
    batch_size: int = 32
    learn_rate: float = 1e-3


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        latent_dim: int,
        hidden_dims: List = None,
        img_size: int = None,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.hidden_dims = hidden_dims
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bottleneck_size = img_size // (2 ** len(hidden_dims))

        # 64, 32, 16, 8, 4, 2

        modules = []
        in_channels += 1  # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(self.get_conv_block(in_channels, h_dim, is_transpose=False))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.bottleneck_size ** 2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.bottleneck_size ** 2, latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(
            latent_dim + num_classes, hidden_dims[-1] * self.bottleneck_size ** 2
        )
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                self.get_conv_block(
                    hidden_dims[i], hidden_dims[i + 1], is_transpose=True
                ),
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            self.get_conv_block(hidden_dims[-1], hidden_dims[-1], is_transpose=True),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    @staticmethod
    def get_conv_block(
        size_in: int, size_out: int, is_transpose: bool
    ) -> nn.Sequential:
        if not is_transpose:
            conv = nn.Conv2d(size_in, size_out, kernel_size=3, stride=2, padding=1,)
        else:
            conv = nn.ConvTranspose2d(
                size_in, size_out, kernel_size=3, stride=2, padding=1, output_padding=1,
            )
        return nn.Sequential(conv, nn.BatchNorm2d(size_out), nn.LeakyReLU(),)

    def encode(self, inputs: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param inputs: (torch.Tensor) Input tensor to encoder [N x C x H x W]
        :return: (torch.Tensor) List of latent codes
        """
        result = self.encoder(inputs)
        # print(dict(result=result.shape))
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(
            -1, self.hidden_dims[0], self.bottleneck_size, self.bottleneck_size
        )
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (torch.Tensor) Mean of the latent Gaussian
        :param logvar: (torch.Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inputs: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        y = kwargs["labels"].float()
        embedded_class = self.embed_class(y)
        embedded_class = embedded_class.view(
            -1, self.img_size, self.img_size
        ).unsqueeze(1)
        embedded_input = self.embed_data(inputs)

        x = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(x)

        z = self.reparameterize(mu, log_var)

        z = torch.cat([z, y], dim=1)
        return [self.decode(z), inputs, mu, log_var]

    @staticmethod
    def loss_function(*args) -> dict:
        recons = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        kld_weight = 1.0
        # kld_weight = 0.0

        recons_loss = F.mse_loss(recons, inputs)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}

    def sample(
        self, num_samples: int, current_device: torch.device, **kwargs
    ) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (torch.Tensor)
        """
        y = kwargs["labels"].float()
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        z = torch.cat([z, y], dim=1)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (torch.Tensor) [B x C x H x W]
        :return: (torch.Tensor) [B x C x H x W]
        """

        return self.forward(x, **kwargs)[0]


def get_loader(config: Config, train: bool) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = SmallImageDataset(train=train, transform=transform)
    dataset.plot_samples()
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=train, num_workers=4
    )
    return loader


class Experiment(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.model = self.get_model()
        self.curr_device = torch.device("cpu")
        self.batch_train = next(iter(self.train_dataloader()))
        self.batch_val = next(iter(self.val_dataloader()))
        self.sample_dir = Path("samples")
        if self.sample_dir.exists():
            shutil.rmtree(self.sample_dir)
        self.sample_dir.mkdir()

    def get_model(self) -> ConditionalVAE:
        config = self.config
        model_class = dict(cvae=ConditionalVAE)[config.model_name]
        model = model_class(
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            img_size=config.image_size,
        )
        return model

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        losses = self.model.loss_function(
            *results,
            # M_N=self.params["batch_size"] / self.num_train_imgs,
            # optimizer_idx=optimizer_idx,
            # batch_idx=batch_idx,
        )

        log = {key: val.item() for key, val in losses.items()}
        return dict(loss=losses["loss"], log=log)

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            # M_N=self.params["batch_size"] / self.num_val_imgs,
            # optimizer_idx=optimizer_idx,
            # batch_idx=batch_idx,
        )

        return val_loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss": avg_loss}

        if self.current_epoch % 10 == 0:
            self.sample_images(train=True)
            self.sample_images(train=False)

        assert isinstance(avg_loss, torch.Tensor)
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def sample_images(self, train: bool):
        # Get sample reconstruction image
        batch = self.batch_train if train else self.batch_val
        test_input, test_label = batch
        # test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels=test_label)

        data_split = "train" if train else "val"
        s_epoch = str(self.current_epoch).zfill(3)
        path = self.sample_dir / data_split / f"{s_epoch}.png"
        path.parent.mkdir(exist_ok=True)
        save_image(recons.data, str(path), nrow=12, normalize=True)

    def train_dataloader(self) -> DataLoader:
        return get_loader(self.config, train=True)

    def val_dataloader(self) -> DataLoader:
        return get_loader(self.config, train=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learn_rate)


def main():
    config = Config()
    experiment = Experiment(config)
    trainer = pl.Trainer(gpus=1)
    trainer.fit(experiment)


if __name__ == "__main__":
    main()
