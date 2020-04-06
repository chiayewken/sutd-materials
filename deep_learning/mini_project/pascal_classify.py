import argparse
import pathlib

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision
import torchvision.datasets.utils
import tqdm
from PIL import Image
from sklearn import metrics

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dict(device=device))
    return device


class Splits:
    train = "train"
    val = "val"
    trainval = "trainval"

    @classmethod
    def check_valid(cls, data_split: str) -> bool:
        return data_split in {cls.train, cls.val, cls.trainval}


class PascalMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_split, image_size):
        assert Splits.check_valid(data_split)
        self.name = self.__class__.__name__
        self.data_split = data_split
        self.data_root = pathlib.Path(data_root)
        t = torchvision.transforms
        self.transform = t.Compose(
            [
                t.CenterCrop(image_size),
                t.RandomHorizontalFlip(),
                t.ToTensor(),
                t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.extract_dir = self.download()
        self.process_data()
        self.image_paths, self.targets = self.process_data()
        print(self)
        self.run_check()

    def run_check(self):
        for i in range(3):
            x, y = self[i]
            print(dict(x=x.shape, y=y, path=self.image_paths[i]))

    def __repr__(self):
        info = dict(
            data_split=self.data_split,
            root=self.data_root,
            len=len(self),
            targets=self.targets.shape,
        )
        return str({self.name: info})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        path = self.image_paths[i]
        image = Image.open(path)
        image = self.transform(image)
        return image, self.targets[i]

    @staticmethod
    def get_labels():
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def download(self):
        extract_dir = self.data_root / "VOCdevkit"
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/VOCtrainval_11-May-2012.tar"
        filename = url.split("/")[-1]
        compressed = self.data_root / filename

        if not extract_dir.exists():
            torchvision.datasets.utils.download_url(url, self.data_root, filename)
            assert compressed.exists()
            torchvision.datasets.utils.extract_archive(str(compressed))

        return extract_dir

    def process_data(self):
        main_dir = self.extract_dir / "VOC2012"
        image_dir = main_dir / "JPEGImages"
        set_dir = main_dir / "ImageSets" / "Main"
        assert image_dir.exists()
        assert set_dir.exists()

        name2labels = {}
        label2idx = {label: i for i, label in enumerate(self.get_labels())}
        for label in label2idx.keys():
            with open(set_dir / f"{label}_{self.data_split}.txt") as f:
                for line in f:
                    image_name, indicator = line.strip().split()
                    is_positive = {"1": True, "-1": False, "0": False}[indicator]
                    if is_positive:
                        if image_name not in name2labels.keys():
                            name2labels[image_name] = []
                        name2labels[image_name].append(label)

        image_paths = []
        targets = np.zeros(shape=(len(name2labels), len(label2idx)))
        for i_image, (image_name, pos_labels) in enumerate(name2labels.items()):
            path = (image_dir / image_name).with_suffix(".jpg")
            assert path.exists()
            image_paths.append(path)
            for label in pos_labels:
                targets[i_image, label2idx[label]] = 1

        assert len(image_paths) == targets.shape[0]
        return image_paths, torch.from_numpy(targets).float()


class HyperParams(argparse.Namespace):
    """Container for hyper-parameters, facilitate pl.loggers, code completion"""

    def __init__(
        self,
        data_root="temp",
        image_size=224,
        learn_rate=1e-4,
        batch_size=32,
        test_run=False,
        pretrained=True,
        model="resnet18",
        version=0,
        **kwargs,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.test_run = test_run
        self.pretrained = pretrained
        self.model = model
        self.version = version

        kwargs.update(vars(self))
        self.check_kwargs(**kwargs)
        super().__init__(**kwargs)
        print(self)

    def check_kwargs(self, **kwargs):
        for k in kwargs.keys():
            assert hasattr(self, k)


class PascalClassifySystem(pl.LightningModule):
    def __init__(self, hparams: HyperParams):
        super().__init__()
        self.hparams = hparams
        self.name = self.__class__.__name__
        self.output_dir = pathlib.Path(self.hparams.data_root) / self.name
        self.checkpoint_dir = self.get_checkpoint_dir()
        self.criterion = torch.nn.BCELoss()
        self.num_labels = len(self.get_data(Splits.train)[0].get_labels())
        self.net = self.get_model()
        print(self)

    def get_checkpoint_dir(self):
        return (
            self.output_dir
            / "lightning_logs"
            / f"version_{self.hparams.version}"
            / "checkpoints"
        )

    def get_model(self):
        selector = {
            "resnet18": torchvision.models.resnet18,
            "resnet50": torchvision.models.resnet50,
        }
        model_fn = selector[self.hparams.model]
        net = model_fn(pretrained=self.hparams.pretrained)
        net.fc = torch.nn.Linear(net.fc.in_features, self.num_labels)
        return net

    def __repr__(self):
        info = dict(
            num_labels=self.num_labels,
            output_dir=self.output_dir,
            net=self.net.__class__.__name__,
        )
        return str({self.name: info})

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

    def get_data(self, data_split):
        dataset = PascalMultiLabelDataset(
            self.output_dir, data_split, self.hparams.image_size
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=(data_split == Splits.train),
            num_workers=2,
        )
        return dataset, loader

    def training_step(self, batch, i_batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        log = dict(train_loss=loss)
        return dict(loss=loss, log=log)

    def validation_step(self, batch, i_batch):
        return self.training_step(batch, i_batch)

    def validation_epoch_end(self, outputs: list):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log = dict(val_loss=loss)
        return dict(log=log, progress_bar=log)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.hparams.learn_rate)

    def train_dataloader(self):
        return self.get_data(Splits.train)[1]

    def val_dataloader(self):
        return self.get_data(Splits.val)[1]

    def run_train(self):
        if self.checkpoint_dir.exists():
            return

        trainer = pl.Trainer(
            early_stop_callback=pl.callbacks.EarlyStopping(patience=3, verbose=True),
            gpus=(1 if torch.cuda.is_available() else None),
            fast_dev_run=self.hparams.test_run,
            default_save_path=self.output_dir,
        )
        trainer.fit(self)

    def load_latest_checkpoint(self):
        paths = list(self.checkpoint_dir.iterdir())
        paths = sorted(paths, key=lambda x: str(x))
        assert all([p.suffix == ".ckpt" for p in paths])
        path_latest = paths[-1]
        print(dict(load_latest_checkpoint=path_latest))
        loaded_system = self.load_from_checkpoint(str(path_latest))
        assert loaded_system.hparams == self.hparams
        self.load_state_dict(loaded_system.state_dict())

    def plot_top_images(
        self, dataset, outputs: torch.Tensor, num=5, reverse=False, labels=None
    ):
        labels_orig = dataset.get_labels()
        if labels is None:
            labels = labels_orig

        num_samples, num_labels = outputs.shape
        assert num_samples == len(dataset)

        ranks = torch.argsort(outputs, dim=0, descending=(not reverse))
        images = []
        for lab in labels:
            i_label = labels_orig.index(lab)
            for i_image in ranks[:num, i_label]:
                x, y = dataset[i_image]
                images.append(x)
        path = self.checkpoint_dir / f"plot_rank_{num}_reverse_{reverse}.png"
        torchvision.utils.save_image(images, str(path), nrow=num, normalize=True)
        print(path, labels)

    def get_eval_outputs(self, loader):
        path_cache = self.checkpoint_dir / "eval_outputs.pt"
        if not path_cache.exists():
            device = get_device()
            self.load_latest_checkpoint()
            self.to(device)
            self.eval()

            outputs_history = []
            targets_history = []
            with torch.no_grad():
                for x, y in tqdm.tqdm(loader):
                    targets_history.append(y)
                    outputs_history.append(self.forward(x.to(device)).cpu())
            outputs = torch.cat(outputs_history, dim=0)
            targets = torch.cat(targets_history, dim=0)
            torch.save((outputs, targets), str(path_cache))

        return torch.load(str(path_cache))

    def plot_tail_acc(self, df_results: pd.DataFrame):
        prefix = "tail_acc_"
        cols = [c for c in df_results.columns if prefix in c]
        thresholds = np.array([float(c.strip(prefix)) for c in cols])
        values = df_results[cols].mean(axis=0).values.squeeze()
        argsort = np.argsort(thresholds)

        plt.clf()
        plt.plot(thresholds[argsort], values[argsort])
        plt.xlabel("thresholds")
        plt.ylabel("tail accuracy")
        plt.xticks(thresholds)
        path = self.checkpoint_dir / "tail_acc_plot.png"
        print(path)
        plt.savefig(str(path))

    def run_eval(self):
        dataset, loader = self.get_data(Splits.val)
        outputs, targets = self.get_eval_outputs(loader)
        results = []
        thresholds = np.linspace(0.5, 0.95, 10).round(3)
        labels = dataset.get_labels()

        for i, label in enumerate(labels):
            _outputs = outputs[:, i]
            _targets = targets[:, i]
            loss = self.criterion(_outputs, _targets).item()
            precision = metrics.average_precision_score(
                y_true=_targets.numpy(), y_score=_outputs.numpy()
            )
            res = dict(label=label, precision=precision, loss=loss)
            for thresh in thresholds:
                tail_acc = tail_acc_score(_outputs, _targets, thresh)
                res[f"tail_acc_{thresh}"] = tail_acc
            results.append(res)
        df = pd.DataFrame(results)
        print_full_dataframe(df)
        print("Averaged:", df.drop(columns=["label"]).mean(axis=0))

        labels_plot = [labels[i] for i in (18, 1, 19, 8, 10)]  # Fixed random
        self.plot_tail_acc(df)
        self.plot_top_images(dataset, outputs, labels=labels_plot)
        self.plot_top_images(dataset, outputs, labels=labels_plot, reverse=True)


def print_full_dataframe(df):
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)


def tail_acc_score(outputs: torch.Tensor, targets: torch.Tensor, threshold=0.5):
    assert outputs.shape == targets.shape
    assert outputs.dtype == targets.dtype
    assert outputs.ndim == 1
    mask = outputs > threshold
    targets = targets.masked_select(mask)
    preds = (outputs.masked_select(mask) > threshold).float()
    num_correct = targets.eq(preds).float().sum()
    total = targets.shape[0]
    return (num_correct / total).item()


def main(test_run=False):
    experiments = [
        HyperParams(version=0, test_run=test_run, pretrained=False),
        HyperParams(version=1, test_run=test_run, pretrained=True),
        HyperParams(version=2, test_run=test_run, pretrained=True, model="resnet50"),
    ]
    for e in experiments:
        system = PascalClassifySystem(e)
        system.run_train()
        system.run_eval()


if __name__ == "__main__":
    fire.Fire(main)
