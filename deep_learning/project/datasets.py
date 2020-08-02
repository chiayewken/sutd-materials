import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torchmeta
import torchvision
from sklearn import metrics, linear_model
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url

from embedding import WordEmbedder, SentenceBERT
from utils import shuffle_multi_split, HyperParams


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

    @classmethod
    def get_all(cls):
        return cls.train, cls.val, cls.test


class OmniglotMetaLoader(torchmeta.utils.data.BatchMetaDataLoader):
    def __init__(self, hp: HyperParams, data_split: str):
        assert Splits.check(data_split)
        self.data_split = data_split
        self.params = hp
        self.dataset, self.split_dataset = self.get_dataset()
        super().__init__(self.split_dataset, hp.bs_outer, num_workers=2)
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
            num_test_per_class=self.params.num_shots,
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
        print(dict(meta_split=self.meta_split, labels_keep=labels_keep))
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
        self,
        dataset: Dataset,
        data_split: str,
        hp: HyperParams,
        transform=None,
        num_workers=0,
    ):
        assert Splits.check(data_split)
        self.transform = transform
        self.ds_orig = dataset
        self.ds_class = ClassDataset(self.ds_orig, data_split, transform)
        self.ds_meta = torchmeta.utils.data.CombinationMetaDataset(
            dataset=self.ds_class,
            num_classes_per_task=hp.num_ways,
            target_transform=torchmeta.transforms.Categorical(hp.num_ways),
        )
        self.ds_split = torchmeta.transforms.ClassSplitter(
            self.ds_meta,
            shuffle=(data_split == Splits.train),
            # num_train_per_class=hp.num_shots,
            # num_test_per_class=hp.num_shots,
            num_samples_per_class={s: hp.num_shots for s in Splits.get_all()},
        )
        super().__init__(
            dataset=self.ds_split,
            batch_size=hp.bs_outer,
            shuffle=(data_split == Splits.train),
            num_workers=num_workers,
        )


class MetaTask:
    def __init__(self, task: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        self.train = task[Splits.train]
        self.val = task[Splits.val]
        self.test = task[Splits.test]


class MetaBatch:
    """Convenience class to process batches of episodes/tasks"""

    def __init__(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], device: torch.device,
    ):
        self.device = device
        self.batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {
            s: (x.to(device), y.to(device)) for s, (x, y) in batch.items()
        }

    def get_tasks(self,) -> List[MetaTask]:
        batch_size: int = self.batch[Splits.train][0].shape[0]
        return [
            MetaTask({s: (x[i], y[i]) for s, (x, y) in self.batch.items()})
            for i in range(batch_size)
        ]


class IntentDataset(Dataset):
    """Intent text classification"""

    def __init__(self, root: str, remove_oos_orig=True):
        self.remove_oos_orig = remove_oos_orig
        self.root = Path(root)
        self.data_orig = self.prepare_data()
        self.texts, self.labels = self.process_data()
        self.unique_labels = sorted(set(self.labels))

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

    def __getitem__(self, i: int) -> Tuple[str, str]:
        return self.texts[i], self.labels[i]

    def __len__(self) -> int:
        assert len(self.texts) == len(self.labels)
        return len(self.texts)


class IntentEmbedBertDataset(IntentDataset):
    """Use pre-trained BERT language model to obtain sentence embeddings as pre-processing"""

    def __init__(
        self, root: str, remove_oos_orig=True, try_embeds=False, try_label_embeds=False
    ):
        super().__init__(root, remove_oos_orig)
        self.embedder = SentenceBERT()
        self.embeds = self.embedder.embed_texts(self.texts, cache_name="texts")
        self.unique_label_embeds = self.embed_labels()
        if try_embeds:
            self.try_embeds_fit_simple_classifier()
        if try_label_embeds:
            self.test_zero_shot_label_embeds()

    def __getitem__(self, i: int) -> Tuple[np.ndarray, str]:
        return self.embeds[i], self.labels[i]

    def embed_labels(self):
        labels = [label.replace("_", " ") for label in self.unique_labels]
        return self.embedder.embed_texts(labels, cache_name="labels")

    def test_zero_shot_label_embeds(self):
        dists = metrics.pairwise.euclidean_distances(
            self.embeds, self.unique_label_embeds
        )
        pred_indices = np.argmin(dists, axis=-1)
        labels = np.array(self.unique_labels)
        preds = labels[pred_indices]
        print(metrics.classification_report(self.labels, preds))

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


class IntentEmbedWordDataset(IntentDataset):
    def __init__(self, root: str, remove_oos_orig=True):
        super().__init__(root, remove_oos_orig)
        self.embedder = WordEmbedder()
        self.embeds = self.embedder.embed_texts(self.texts)
        self.refine_embeds()
        print(dict(embeds=self.embeds.shape))

    def refine_embeds(self, maxlen_percentile=95):
        lengths = [len(_) for _ in self.embeds]
        lengths = sorted(lengths)
        maxlen = lengths[len(lengths) * maxlen_percentile // 100]

        embeds = np.zeros(shape=(len(lengths), maxlen, self.embedder.embed_size))
        for i, row in enumerate(self.embeds):
            row = row[:maxlen]
            embeds[i, -len(row) :, :] = row
        self.embeds = embeds.astype(np.float32)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, str]:
        return self.embeds[i], self.labels[i]


class IntentEmbedWordMeanDataset(IntentEmbedWordDataset):
    def __init__(self, root: str, remove_oos_orig=True):
        super().__init__(root, remove_oos_orig)

    def refine_embeds(self, dummy_argument=0):
        self.embeds = np.stack([np.mean(e, axis=0) for e in self.embeds])
        self.embeds = self.embeds.astype(np.float32)


class IntentWordIndicesDataset(IntentDataset):
    def __init__(self, root: str, remove_oos_orig=True):
        super().__init__(root, remove_oos_orig)
        self.embedder = WordEmbedder()
        self.embedder.fit_texts(self.texts)
        self.indices = np.array([self.embedder.encode(t) for t in self.texts])
        self.indices = self.refine()

    def refine(self, maxlen_percentile=95) -> np.ndarray:
        lengths = [len(_) for _ in self.indices]
        lengths = sorted(lengths)
        maxlen = lengths[len(lengths) * maxlen_percentile // 100]

        pad = self.embedder.word2i["empty"]
        indices = np.full(shape=(len(lengths), maxlen), fill_value=pad, dtype=np.int64)
        for i, row in enumerate(self.indices):
            row = row[:maxlen]
            indices[i, -len(row) :] = row

        assert np.min(indices) >= 0
        assert np.max(indices) < len(self.embedder.vocab)
        info = dict(
            shape=indices.shape,
            dtype=indices.dtype,
            min=np.min(indices),
            max=np.max(indices),
            unique=len(np.unique(indices)),
        )
        print(dict(indices=info))
        return indices

    def __getitem__(self, i: int) -> Tuple[np.ndarray, str]:
        return self.indices[i], self.labels[i]


class IntentEmbedBertMetaLoader(MetaLoader):
    def __init__(self, hp: HyperParams, data_split: str, do_test=False):
        dataset = IntentEmbedBertDataset(
            root=hp.root, try_embeds=do_test, try_label_embeds=do_test
        )
        self.embed_size = dataset.embedder.embed_size
        super().__init__(dataset, data_split, hp)


class IntentEmbedWordMetaLoader(MetaLoader):
    def __init__(self, hp: HyperParams, data_split: str):
        dataset = IntentEmbedWordDataset(root=hp.root)
        self.embed_size = dataset.embedder.embed_size
        super().__init__(dataset, data_split, hp)


class IntentEmbedWordMeanMetaLoader(MetaLoader):
    def __init__(self, hp: HyperParams, data_split: str):
        dataset = IntentEmbedWordMeanDataset(root=hp.root)
        self.embed_size = dataset.embedder.embed_size
        super().__init__(dataset, data_split, hp)


class IntentWordIndicesMetaLoader(MetaLoader):
    def __init__(self, hp: HyperParams, data_split: str):
        dataset = IntentWordIndicesDataset(root=hp.root)
        self.embed_size = dataset.embedder.embed_size
        self.vocab_size = len(dataset.embedder.vocab)
        self.embeds = dataset.embedder.embeds
        super().__init__(dataset, data_split, hp)


def run_test():
    # loader = IntentWordIndicesMetaLoader(HyperParams(), data_split=Splits.train)
    # print(next(iter(loader)).keys())
    # IntentEmbedWordMetaLoader(HyperParams(), data_split=Splits.train)
    IntentEmbedBertMetaLoader(HyperParams(), data_split=Splits.train, do_test=True)
    # OmniglotMetaLoader(HyperParams(), data_split=Splits.train)


if __name__ == "__main__":
    run_test()
