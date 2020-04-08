from typing import List

import torch
from sklearn import model_selection


class HyperParams:
    def __init__(
        self,
        root="data",
        lr=1e-3,
        bs=32,
        steps_per_epoch=1000,
        epochs=100,
        model="lstm",
        n_hidden=128,
        n_layers=1,
        dropout=0.0,
        tie_embed_weights=True,
        seq_len=32,
        dev_run=False,
        verbose=True,
    ):
        self.root = root
        self.lr = lr
        self.bs = bs
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.n_hidden = n_hidden
        self.model = model
        self.n_layers = n_layers
        self.dropout = dropout
        self.tie_embed_weights = tie_embed_weights
        self.seq_len = seq_len
        self.dev_run = dev_run
        self.verbose = verbose

        if self.verbose:
            print(self)

    def __eq__(self, other):
        assert isinstance(other, HyperParams)
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str({self.__class__.__name__: self.__dict__})


class Splits:
    train = "train"
    val = "val"
    test = "test"

    @classmethod
    def check_valid(cls, x: str) -> bool:
        return x in {cls.train, cls.val, cls.test}


class Vocab:
    pad = "<pad>"
    start = "<start>"
    end = "<end>"
    unk = "<unk>"

    def __init__(self, items: List[str], use_special_tokens=True):
        self.special = []
        if use_special_tokens:
            self.special = [self.pad, self.start, self.end, self.unk]

        unique = self.special + sorted(set(items))
        self.stoi = {s: i for i, s in enumerate(unique)}
        self.itos = {i: s for i, s in enumerate(unique)}

    def __len__(self) -> int:
        assert len(self.stoi) == len(self.itos)
        return len(self.stoi)

    def encode(self, items: List[str]) -> List[int]:
        return [self.stoi.get(s, self.stoi[self.unk]) for s in items]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.itos[i] for i in indices]


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


def get_device(verbose=True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(dict(device=device))
    return device


class Sampler:
    @staticmethod
    def softmax(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=-1)

    @classmethod
    def sample(
        cls, logits: torch.Tensor, thresholds: torch.Tensor = None
    ) -> torch.Tensor:
        if thresholds is not None:
            assert logits.ndim == thresholds.ndim
            logits = logits.masked_fill(logits < thresholds, -1e9)
        return torch.multinomial(cls.softmax(logits), num_samples=1)

    @classmethod
    def temperature(cls, logits: torch.Tensor, t=0.5) -> torch.Tensor:
        return cls.sample(logits / t)

    @classmethod
    def top_k(cls, logits: torch.Tensor, k=4) -> torch.Tensor:
        _sorted, _ = torch.sort(logits, dim=-1, descending=True)
        thresholds = _sorted[:, [k]]
        return cls.sample(logits, thresholds)

    @classmethod
    def top_p(cls, logits: torch.Tensor, p=0.99) -> torch.Tensor:
        _sorted, _ = torch.sort(logits, dim=-1, descending=True)
        cumsum = torch.cumsum(cls.softmax(_sorted), dim=-1)
        indices = (cumsum < p).float().sum(dim=-1, keepdim=True).long()
        thresholds = logits.gather(dim=-1, index=indices)
        return cls.sample(logits, thresholds)
