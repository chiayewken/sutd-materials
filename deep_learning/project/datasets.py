import json
from pathlib import Path
from typing import Tuple, List, Dict, Iterable

import numpy as np
import pandas as pd
import sentence_transformers
import torch.utils.data
import torchmeta
import torchvision
from sklearn import metrics, linear_model
from torchvision.datasets.utils import download_url, extract_archive

import utils


class SentenceBERT(sentence_transformers.SentenceTransformer):
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.dir_model = self.download()
        self.size_embed = 768
        super().__init__(str(self.dir_model), device=utils.get_device())

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
        assert embeds.shape == (len(texts), self.size_embed)
        return embeds


class Splits:
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def check(cls, data_split: str) -> bool:
        return data_split in {cls.train, cls.val, cls.test}


class MetaDataParams:
    """Hyperparameters for meta learning data"""

    def __init__(
        self, root="temp", bs=10, num_ways=5, num_shots=5, num_shots_test=5,
    ):
        self.root = root
        self.bs = bs
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test
        print(vars(self))


class OmniglotMetaLoader(torchmeta.utils.data.BatchMetaDataLoader):
    def __init__(self, params: MetaDataParams, data_split: str):
        assert Splits.check(data_split)
        self.data_split = data_split
        self.params = params
        self.dataset, self.split_dataset = self.get_dataset()
        super().__init__(self.split_dataset, params.bs, num_workers=2)
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


class IntentDataset(torch.utils.data.Dataset):
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


class IntentClassDataset(torchmeta.utils.data.ClassDataset):
    def __init__(self, orig_dataset: IntentDataset, meta_split: str, transform=None):
        super().__init__(meta_split=meta_split)
        self.orig_dataset = orig_dataset
        self.transform = transform
        self.label2texts = self.process_data()
        self.labels = sorted(self.label2texts.keys())

    def process_data(self) -> Dict[str, List[str]]:
        unique_labels = sorted(set(self.orig_dataset.labels))
        splits = {Splits.train: 0.8, Splits.val: 0.1, Splits.test: 0.1}
        labels_split = utils.shuffle_multi_split(
            items=unique_labels, fractions=list(splits.values())
        )
        i_split = list(splits.keys()).index(self.meta_split)
        labels = labels_split[i_split]
        label2texts = {label: [] for label in labels}
        for i in range(len(self.orig_dataset)):
            text, label = self.orig_dataset[i]
            if label in label2texts.keys():
                label2texts[label].append(text)
        return label2texts

    def __getitem__(self, i) -> SingleLabelDataset:
        label = self.labels[i]
        return SingleLabelDataset(
            index=i,
            data=self.label2texts[label],
            label=label,
            transform=self.get_transform(i, self.transform),
            target_transform=self.get_target_transform(i),
        )

    @property
    def num_classes(self) -> int:
        return len(self.labels)


class IntentMetaLoader(torchmeta.utils.data.BatchMetaDataLoader):
    def __init__(self, params: MetaDataParams, data_split: str, try_embeds=False):
        assert Splits.check(data_split)
        self.data_split = data_split
        self.params = params
        self.orig_dataset = IntentDataset(self.params.root)
        self.embedder = SentenceBERT()
        self.transform = self.get_transform(self.orig_dataset)
        self.class_dataset, self.meta_dataset, self.split_dataset = self.get_datasets()
        super().__init__(
            dataset=self.split_dataset,
            batch_size=params.bs,
            shuffle=(self.data_split == Splits.train),
            num_workers=2,
        )
        if try_embeds:
            self.try_embeds_fit_simple_classifier()

    def get_transform(self, dataset: IntentDataset):
        texts = dataset.texts
        embeds = self.embedder.embed_texts(texts)
        text2embed = {texts[i]: embeds[i] for i in range(len(texts))}
        return lambda x: text2embed[x]

    def get_datasets(self):
        class_dataset = IntentClassDataset(
            self.orig_dataset, self.data_split, transform=self.transform
        )
        meta_dataset = torchmeta.utils.data.CombinationMetaDataset(
            dataset=class_dataset,
            num_classes_per_task=self.params.num_ways,
            target_transform=torchmeta.transforms.Categorical(self.params.num_ways),
        )
        split_dataset = torchmeta.transforms.ClassSplitter(
            meta_dataset,
            shuffle=(self.data_split == Splits.train),
            num_train_per_class=self.params.num_shots,
            num_test_per_class=self.params.num_shots_test,
        )
        return class_dataset, meta_dataset, split_dataset

    def try_embeds_fit_simple_classifier(self):
        embeds = np.stack([self.transform(text) for text in self.orig_dataset.texts])
        labels = np.array(self.orig_dataset.labels)
        train, val, test = utils.shuffle_multi_split(list(range(len(embeds))))
        model = linear_model.RidgeClassifier()
        model.fit(embeds[train], labels[train])
        print(metrics.classification_report(labels[val], model.predict(embeds[val])))


class MetaBatch:
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


def main():
    IntentMetaLoader(MetaDataParams(), data_split=Splits.train)
    OmniglotMetaLoader(MetaDataParams(), data_split=Splits.train)


if __name__ == "__main__":
    main()
