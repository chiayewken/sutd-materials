import json
import pathlib
from typing import List, Dict, Tuple

import fire
import numpy as np
import torch
import torch.utils.data
import tqdm
from sklearn import model_selection, feature_extraction, linear_model, pipeline, metrics
from torchvision.datasets.utils import download_and_extract_archive


class Splits:
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def validate(cls, x: str) -> bool:
        return x in {cls.train, cls.val, cls.test}


class Vocab:
    def __init__(self, items: List[str]):
        self.s_pad = "<pad>"
        self.i_pad = 0
        unique = [self.s_pad] + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(unique)}
        self.itos = {i: s for i, s in enumerate(unique)}
        assert self.stoi[self.s_pad] == self.i_pad
        print(dict(vocab=len(self)))

    def __len__(self) -> int:
        assert len(self.stoi) == len(self.itos)
        return len(self.stoi)

    def encode(self, items: List[str]) -> List[int]:
        return [self.stoi[s] for s in items]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.itos[i] for i in indices]


class NameLanguageClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, data_split: str):
        assert Splits.validate(data_split)
        self.data_split = data_split
        self.splits = {Splits.train: 0.8, Splits.val: 0.1, Splits.test: 0.1}
        self.root = pathlib.Path(root)
        self.root = self.download()
        self.raw = self.get_raw_data()
        self.vocab_text, self.vocab_label = self.build_vocab()
        self.texts, self.labels = self.get_split_data()
        self.inputs, self.targets = self.get_encoded_data()

    def download(self) -> pathlib.Path:
        url = "https://download.pytorch.org/tutorial/data.zip"
        data_dir = self.root / "data"

        if not data_dir.exists():
            download_and_extract_archive(url, str(self.root))
        assert data_dir.exists()
        return data_dir

    def get_raw_data(self) -> Tuple[List[str], List[str]]:
        texts = []
        labels = []

        for path_lang in (self.root / "names").glob("*.txt"):
            lang = path_lang.stem
            with open(str(path_lang)) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
                        labels.append(lang)
        assert len(texts) == len(labels)
        print(dict(get_raw_data=len(texts)))
        return texts, labels

    def build_vocab(self) -> Tuple[Vocab, Vocab]:
        texts, labels = self.raw
        vocab_text = Vocab(list("".join(texts)))
        vocab_label = Vocab(labels)
        return vocab_text, vocab_label

    @staticmethod
    def get_max_sequence_length(sequences: List[list], percentile=95) -> int:
        lengths = [len(_) for _ in sequences]
        max_len = int(np.percentile(lengths, percentile))
        print(dict(get_max_sequence_length=max_len))
        return max_len

    def get_split_data(self) -> Tuple[List[str], List[str]]:
        texts, labels = self.raw
        indices_all = list(range(len(texts)))
        indices_split = shuffle_multi_split(indices_all, list(self.splits.values()))
        split_keys = list(self.splits.keys())
        split2indices = {s: i for s, i in zip(split_keys, indices_split)}
        indices = split2indices[self.data_split]

        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        assert len(texts) == len(labels)
        print(dict(get_split_data=len(texts)))
        return texts, labels

    def get_encoded_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = torch.from_numpy(np.array(self.vocab_label.encode(self.labels)))
        num_samples = len(self.texts)
        encoded = [self.vocab_text.encode(list(t)) for t in self.texts]
        max_len = self.get_max_sequence_length(encoded)
        inputs_numpy = np.full(
            shape=(num_samples, max_len), fill_value=self.vocab_text.i_pad
        )
        for i in range(num_samples):
            sequence = encoded[i][:max_len]
            inputs_numpy[i, : len(sequence)] = sequence
        inputs = torch.from_numpy(inputs_numpy).type(torch.long)
        assert inputs.shape[0] == targets.shape[0]
        print(dict(inputs=inputs.shape, targets=targets.shape))
        return inputs, targets

    def show_samples(self, num=10):
        print(dict(show_samples=num))
        indices = np.random.choice(len(self), size=num, replace=False)
        for i in indices:
            inputs, targets = self[i]
            text = "".join(self.vocab_text.decode(inputs.numpy()))
            label = self.vocab_label.decode([targets.item()])
            print(dict(text=text, label=label, raw_inputs=inputs, raw_targets=targets))

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[i], self.targets[i]

    def __len__(self) -> int:
        return self.inputs.shape[0]


def normalize(items: list) -> list:
    total = sum(items)
    return [item / total for item in items]


def shuffle_multi_split(
    items: list, fractions=(0.8, 0.1, 0.1), seed=42, eps=1e-6
) -> list:
    assert abs(sum(fractions) - 1) < eps
    assert len(fractions) > 0
    if len(fractions) == 1:
        return [items]

    part_first, part_rest = model_selection.train_test_split(
        items, train_size=fractions[0], random_state=seed
    )
    parts_all = [part_first] + shuffle_multi_split(
        part_rest, normalize(fractions[1:]), seed
    )
    assert len(items) == sum(map(len, parts_all))
    return parts_all


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HyperParams:
    def __init__(
        self,
        root="data",
        learn_rate=5e-3,
        bs=10,
        max_epochs=100,
        n_hidden=128,
        rnn="lstm",
        n_layers=1,
        dev_run=False,
    ):
        self.root = root
        self.learn_rate = learn_rate
        self.batch_size = bs
        self.max_epochs = max_epochs
        self.dim_hidden = n_hidden
        self.rnn_type = rnn
        self.rnn_num_layers = n_layers
        self.fast_dev_run = dev_run
        print(self.__class__.__name__, self.__dict__)


class Net(torch.nn.Module):
    def __init__(
        self, num_vocab: int, num_labels: int, hparams: HyperParams, batch_first=True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.embed = torch.nn.Embedding(num_vocab, hparams.dim_hidden)
        rnn_class = dict(lstm=torch.nn.LSTM, gru=torch.nn.GRU)[hparams.rnn_type]
        self.rnn = rnn_class(
            input_size=hparams.dim_hidden,
            hidden_size=hparams.dim_hidden,
            num_layers=hparams.rnn_num_layers,
            batch_first=self.batch_first,
        )
        self.linear = torch.nn.Linear(hparams.dim_hidden, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x, states = self.rnn(x)
        x = torch.mean(x, dim=(1 if self.batch_first else 0))
        x = self.linear(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x


class NameLanguageClassifySystem:
    def __init__(
        self, hparams: HyperParams,
    ):
        self.hparams = hparams
        self.data_splits = [Splits.train, Splits.val, Splits.test]
        self.device = get_device()
        self.datasets = {s: self.get_dataset(s) for s in self.data_splits}
        self.net = Net(
            num_vocab=len(self.datasets[Splits.train].vocab_text),
            num_labels=len(self.datasets[Splits.train].vocab_label),
            hparams=hparams,
        )
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=hparams.learn_rate)
        # self.fit_sklearn_baseline()
        self.datasets[Splits.train].show_samples()

    def get_dataset(self, data_split: str) -> NameLanguageClassifyDataset:
        return NameLanguageClassifyDataset(self.hparams.root, data_split)

    def get_loader(self, data_split: str) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.datasets[data_split],
            batch_size=self.hparams.batch_size,
            shuffle=(data_split == Splits.train),
        )

    def fit_sklearn_baseline(self):
        pipe = pipeline.make_pipeline(
            feature_extraction.text.TfidfVectorizer(
                analyzer="char", ngram_range=(3, 6)
            ),
            linear_model.RidgeClassifier(),
        )
        print(dict(fit_sklearn_baseline=pipe))
        ds_train = self.datasets[Splits.train]
        ds_val = self.datasets[Splits.val]
        pipe.fit(ds_train.texts, ds_train.labels)
        print(metrics.classification_report(ds_val.labels, pipe.predict(ds_val.texts)))

    def run_epoch(self, data_split: str) -> Dict[str, float]:
        training = data_split == Splits.train
        if training:
            self.net.train()
            context_grad = torch.enable_grad
        else:
            self.net.eval()
            context_grad = torch.no_grad

        correct = 0
        total = 0
        losses = []
        with context_grad():
            for inputs, targets in self.get_loader(data_split):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if training:
                    self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
                predicts = torch.argmax(outputs.data, dim=1)
                total += targets.shape[0]
                correct += (predicts == targets).sum().item()
                losses.append(loss.item())
                if training:
                    loss.backward()
                    self.optimizer.step()
                if self.hparams.fast_dev_run:
                    break

        acc = np.round(correct / total, decimals=3)
        loss = np.round(np.mean(losses), decimals=3)
        return dict(loss=loss, acc=acc)

    def train(self, early_stop=True) -> List[dict]:
        self.net.to(get_device())
        loss_best = 1e9

        history = []
        for _ in tqdm.tqdm(range(self.hparams.max_epochs)):
            results = {s: self.run_epoch(s) for s in self.data_splits}
            history.append(results)
            print(results)

            loss_val = results[Splits.val]["loss"]
            if loss_val < loss_best:
                loss_best = loss_val
            elif early_stop:
                break
            if self.hparams.fast_dev_run and len(history) >= 3:
                break

        return history


def main(dev_run=False, path_save_results="results.json"):
    results = []
    for hparams in [
        HyperParams(bs=1, rnn="lstm", n_layers=1, n_hidden=200, dev_run=dev_run),
        HyperParams(bs=10, rnn="lstm", n_layers=1, n_hidden=200, dev_run=dev_run),
        HyperParams(bs=30, rnn="lstm", n_layers=1, n_hidden=200, dev_run=dev_run),
        HyperParams(bs=30, rnn="lstm", n_layers=1, n_hidden=100, dev_run=dev_run),
        HyperParams(bs=30, rnn="lstm", n_layers=2, n_hidden=200, dev_run=dev_run),
        HyperParams(bs=30, rnn="gru", n_layers=1, n_hidden=200, dev_run=dev_run),
    ]:
        system = NameLanguageClassifySystem(hparams)
        results.append(dict(hparams=hparams.__dict__, history=system.train()))
    with open(path_save_results, "w") as f:
        f.write("\n".join([json.dumps(r) for r in results]))


if __name__ == "__main__":
    fire.Fire(main)
