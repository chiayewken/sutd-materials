import json
from pathlib import Path
from typing import Tuple, List, Dict, Iterable

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import torchmeta
import torchvision
from sklearn import metrics, linear_model
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, extract_archive

from utils import get_device, shuffle_multi_split, HyperParams


class SentenceBERT(sentence_transformers.SentenceTransformer):
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.dir_model = self.download()
        self.embed_size = 768
        super().__init__(str(self.dir_model), device=get_device())

    def download(self):
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/bert-base-nli-mean-tokens.zip"
        path_zip = self.cache_dir / Path(url).name
        if not path_zip.exists():
            download_url(url, self.cache_dir, filename=path_zip.name)
        dir_model = self.cache_dir / "bert"
        if not dir_model.exists():
            extract_archive(str(path_zip), str(dir_model))
        return dir_model

    def embed_texts(self, texts: List[str], do_cache=True) -> np.ndarray:
        def get_embeds():
            return np.stack(self.encode(texts, show_progress_bar=True))

        if not do_cache:
            return get_embeds()
        path_cache = self.cache_dir / f"embeds_{len(texts)}.npy"
        if not path_cache.exists():
            np.save(str(path_cache), get_embeds())
        embeds = np.load(str(path_cache))
        assert embeds.shape == (len(texts), self.embed_size)
        return embeds


class Splits:
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def check(cls, data_split: str) -> bool:
        return data_split in {cls.train, cls.val, cls.test}

    @classmethod
    def apply(cls, items, data_split: str, fractions=(0.8, 0.1, 0.1)):
        assert len(fractions) == 3
        assert cls.check(data_split)
        indices_split = shuffle_multi_split(items, fractions)
        splits = [Splits.train, Splits.val, Splits.test]
        return indices_split[splits.index(data_split)]


class OmniglotMetaLoader(torchmeta.utils.data.BatchMetaDataLoader):
    def __init__(self, hparams: HyperParams, data_split: str):
        assert Splits.check(data_split)
        self.data_split = data_split
        self.params = hparams
        self.dataset, self.split_dataset = self.get_dataset()
        super().__init__(self.split_dataset, hparams.bs_outer, num_workers=2)
        self.plot_sample()

    def get_dataset(self, image_size=28):
        dataset = torchmeta.datasets.Omniglot(
            self.params.root,
            num_classes_per_task=self.params.num_ways,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(image_size),
                    torchvision.transforms.ToTensor(),
                ]
            ),
            # Transform the labels to integers (e.g. ("Label_1", "Label_2", ...) to (0, 1, ...))
            target_transform=torchmeta.transforms.Categorical(
                num_classes=self.params.num_ways
            ),
            # Creates new augmented classes  (from Santoro et al., 2016)
            class_augmentations=[torchmeta.transforms.Rotation([90, 180, 270])],
            meta_split=self.data_split,
            download=True,
        )
        split_dataset = torchmeta.transforms.ClassSplitter(
            dataset,
            shuffle=True,
            num_train_per_class=self.params.num_shots,
            num_test_per_class=self.params.num_shots_test,
        )
        return dataset, split_dataset

    def plot_sample(self):
        for batch in self:
            tasks_train = batch["train"]
            images = tasks_train[0][0]
            name = self.__class__.__name__
            root = Path(self.params.root)
            fp = str(root / f"{name}_{self.data_split}_sample.png")
            torchvision.utils.save_image(images, fp, nrow=self.params.num_ways)
            print(fp)
            break


class SingleLabelDataset(torchmeta.utils.data.Dataset):
    """Helper class for meta-learning, consists of samples from only one label"""

    def __init__(self, index, data, label, transform=None, target_transform=None):
        super().__init__(index, transform=transform, target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]
        y = self.label

        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y


class ClassDataset(torchmeta.utils.data.ClassDataset):
    """Helper class for meta-learning, each data "sample" is a SingleLabelDataset"""

    def __init__(self, dataset: Dataset, meta_split: str, transform=None):
        super().__init__(meta_split=meta_split)
        self.transform = transform
        self.full_dataset = dataset
        self.datasets = self.split_datasets()

    def split_datasets(self) -> List[SingleLabelDataset]:
        label2data = {}
        for i in range(len(self.full_dataset)):
            x, y = self.full_dataset[i]
            if y not in label2data.keys():
                label2data[y] = []
            label2data[y].append(x)

        labels_keep = Splits.apply(sorted(label2data.keys()), self.meta_split)
        datasets = []
        for j, (label, data) in enumerate(label2data.items()):
            if label not in labels_keep:
                continue
            datasets.append(SingleLabelDataset(j, data, label, self.transform))
        return datasets

    @property
    def num_classes(self) -> int:
        return len(self.datasets)

    def __getitem__(self, i: int):
        return self.datasets[i]


class MetaLoader(torchmeta.utils.data.BatchMetaDataLoader):
    """Data loader class for meta-learning, samples batches of episodes/tasks"""

    def __init__(
        self, dataset: Dataset, data_split: str, hparams: HyperParams, transform=None
    ):
        assert Splits.check(data_split)
        self.transform = transform
        self.ds_orig = dataset
        self.ds_class = ClassDataset(self.ds_orig, data_split, transform)
        self.ds_meta = torchmeta.utils.data.CombinationMetaDataset(
            dataset=self.ds_class,
            num_classes_per_task=hparams.num_ways,
            target_transform=torchmeta.transforms.Categorical(hparams.num_ways),
        )
        self.ds_split = torchmeta.transforms.ClassSplitter(
            self.ds_meta,
            shuffle=(data_split == Splits.train),
            num_train_per_class=hparams.num_shots,
            num_test_per_class=hparams.num_shots_test,
        )
        super().__init__(
            dataset=self.ds_split,
            batch_size=hparams.bs_outer,
            shuffle=(data_split == Splits.train),
            num_workers=2,
        )


class MetaBatch:
    """Convenience class to process batches of episodes/tasks"""

    def __init__(
        self,
        raw_batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ):
        self.device = device
        self.x_train, self.y_train = self.to_device(raw_batch["train"])
        self.x_test, self.y_test = self.to_device(raw_batch["test"])

    def get_tasks(self,) -> List[List[torch.Tensor]]:
        batch_size = self.x_train.shape[0]
        tensors = [self.x_train, self.y_train, self.x_test, self.y_test]
        return [[t[i] for t in tensors] for i in range(batch_size)]

    def to_device(self, tensors: Iterable[torch.Tensor]) -> List[torch.Tensor]:
        return [t.to(self.device) for t in tensors]


class IntentDataset(Dataset):
    """Intent text classification"""

    def __init__(self, root: str, remove_oos_orig=True):
        self.remove_oos_orig = remove_oos_orig
        self.root = Path(root)
        self.data_orig = self.prepare_data()
        self.texts, self.labels = self.process_data()

    def prepare_data(self) -> pd.DataFrame:
        url = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
        filename = Path(url).name
        path_json = self.root / filename

        if not path_json.exists():
            download_url(url, str(self.root), filename)

        data = []
        with open(path_json) as f:
            raw: dict = json.load(f)
        for type_split, samples in raw.items():
            for text, label in samples:
                data.append(dict(text=text, label=label, split=type_split))
        return pd.DataFrame(data)

    def process_data(self) -> Tuple[List[str], List[str]]:
        df = self.data_orig.copy()
        if self.remove_oos_orig:
            mask_oos = df["split"].apply(lambda x: "oos" in x)
            # df_oos = df[mask_oos]
            df = df[~mask_oos]
        texts = df["text"].tolist()
        labels = df["label"].tolist()
        return texts, labels

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]

    def __len__(self):
        assert len(self.texts) == len(self.labels)
        return len(self.texts)


class IntentEmbedBertDataset(IntentDataset):
    """Use pre-trained BERT language model to obtain sentence embeddings as pre-processing"""

    def __init__(self, root: str, remove_oos_orig=True, try_embeds=False):
        super().__init__(root, remove_oos_orig)
        self.embedder = SentenceBERT()
        self.embeds = self.embedder.embed_texts(self.texts)
        if try_embeds:
            self.try_embeds_fit_simple_classifier()

    def __getitem__(self, i: int) -> Tuple[np.ndarray, str]:
        return self.embeds[i], self.labels[i]

    def try_embeds_fit_simple_classifier(self):
        def get_xy(data_split: str):
            indices = Splits.apply(list(range(len(self))), data_split)
            x = self.embeds
            y = np.array(self.labels)
            return x[indices], y[indices]

        x_train, y_train = get_xy(Splits.train)
        x_val, y_val = get_xy(Splits.val)
        model = linear_model.RidgeClassifier()
        model.fit(x_train, y_train)
        print(metrics.classification_report(y_val, model.predict(x_val)))


class IntentEmbedBertMetaLoader(MetaLoader):
    def __init__(self, hparams: HyperParams, data_split: str):
        dataset = IntentEmbedBertDataset(root=hparams.root)
        self.embed_size = dataset.embedder.embed_size
        super().__init__(dataset, data_split, hparams)


def main():
    # Testing purposes only
    IntentEmbedBertMetaLoader(HyperParams(), data_split=Splits.train)
    OmniglotMetaLoader(HyperParams(), data_split=Splits.train)


if __name__ == "__main__":
    main()
