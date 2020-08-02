import operator
import time
from collections import Iterable
from copy import deepcopy
from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from torch.utils.data import TensorDataset
from tqdm import tqdm


class HyperParams:
    def __init__(
        self,
        root="data",
        num_ways=5,
        num_shots=5,
        loader="embed_bert",
        algo="reptile",
        opt_inner="sgd",
        lr_inner=1e-3,
        lr_outer=1.0,
        steps_inner=5,
        steps_outer=1000,
        bs_inner=10,
        bs_outer=5,
        early_stop=False,
        random_seed=0,
        num_hidden=64,
        num_layers=3,
    ):
        # Data
        self.root = root
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.loader = loader

        # Training
        self.algo = algo
        self.opt_inner = opt_inner
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.steps_inner = steps_inner
        self.steps_outer = steps_outer
        self.bs_inner = bs_inner
        self.bs_outer = bs_outer
        self.early_stop = early_stop
        self.random_seed = random_seed

        # Model
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        print(vars(self))

    def to_string(self):
        return ",".join([f"{k}={v}" for k, v in vars(self).items()])


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # From https://github.com/jakesnell/prototypical-networks/
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x / x.norm(p=2, dim=1, keepdim=True)
    y = y / y.norm(p=2, dim=1, keepdim=True)
    return x.matmul(y.t())


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return cosine_similarity(x, y) * -1 + 1


def accuracy_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds: torch.Tensor
    if logits.shape[-1] == 1:
        preds = logits.sigmoid().round()
    else:
        preds = torch.argmax(logits, dim=-1)
    return (preds == targets).float().mean()


def generate(iterable: Iterable, limit: int = None, show_progress=False):
    counter = 0
    with tqdm(total=limit, disable=(not show_progress)) as progress_bar:
        while True:
            for item in iterable:
                if counter >= limit:
                    return
                progress_bar.update(1)
                counter += 1
                yield item


class RepeatTensorDataset(TensorDataset):
    def __init__(self, *tensors, num_repeat=1):
        super().__init__(*tensors)
        self.num_repeat = num_repeat

    def __len__(self):
        return super().__len__() * self.num_repeat

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        i = i % super().__len__()
        return super().__getitem__(i)


def set_random_state(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(use_gpu=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not use_gpu:
        device = torch.device("cpu")
    print(dict(device=device))
    return device


class EarlyStopSaver:
    def __init__(self, net: torch.nn.Module, patience=3):
        self.net = net
        self.patience = patience
        self.best_loss = 1e9
        self.best_weights: Dict[str, torch.Tensor] = {}
        self.count = 0

    def check_stop(self, loss: float) -> bool:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_weights = deepcopy(self.net.state_dict())
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                return True
        return False

    def load_best(self):
        self.net.load_state_dict(self.best_weights)


class Timer:
    def __init__(self, name, decimals=1):
        self.name = name
        self.decimals = decimals

    def __enter__(self):
        print("Start timer:", self.name)
        self.start = time.time()

    def __exit__(self, _type, value, traceback):
        duration = time.time() - self.start
        duration = round(duration, self.decimals)
        print("Ended timer: {}: {} s".format(self.name, duration))


class MathDict:
    """Allow simple math ops for dicts eg ({key:1} / {key:2}) + 3 = {key:3.5}"""

    def __init__(self, state: dict = None):
        self.state = deepcopy(state)

    def apply(self, other, op):
        if type(other) in {int, float}:
            other = MathDict({k: other for k in self.state.keys()})
        elif type(other) != MathDict:
            raise TypeError(type(other))
        return MathDict({k: op(s, other.state[k]) for k, s in self.state.items()})

    def __add__(self, other):
        return self.apply(other, operator.add)

    def __sub__(self, other):
        return self.apply(other, operator.sub)

    def __mul__(self, other):
        return self.apply(other, operator.mul)

    def __truediv__(self, other):
        return self.apply(other, operator.truediv)

    def __repr__(self):
        return str(self.state)


class LinearDecayLR:
    """Every step, decrease the learning rate linearly until zero"""

    def __init__(self, lr_init, max_steps):
        self.max_steps = max_steps
        self.lr_init = lr_init
        self.plot_schedule()

    def get_lr(self, i_step):
        return self.lr_init * (1 - i_step / self.max_steps)

    def plot_schedule(self):
        steps = list(range(self.max_steps))
        lrs = [self.get_lr(i) for i in steps]
        plt.plot(steps, lrs)
        plt.xlabel("steps")
        plt.ylabel("learn_rate")
        path = f"{self.__class__.__name__}.png"
        plt.savefig(path)
        print(path)


class MetricsTracker:
    """Convenience class to collate dictionaries of metrics eg {loss:1.5, acc:0.5}"""

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.metrics = MathDict()
        self.counter = None
        self.reset()

    def add_key_prefix(self, d: dict):
        return {"_".join([self.prefix, k]): v for k, v in d.items()}

    def reset(self):
        self.metrics = MathDict()
        self.counter = 0

    def store(self, m: dict):
        m = MathDict(m)
        if self.counter == 0:
            self.metrics = m
        else:
            self.metrics = self.metrics + m
        self.counter += 1

    def get_average(self):
        averaged = (self.metrics / self.counter).state
        return self.add_key_prefix(averaged)


def normalize(items: list) -> list:
    total = sum(items)
    return [item / total for item in items]


def shuffle_multi_split(items, fractions=(0.8, 0.1, 0.1), seed=42, eps=1e-6):
    assert abs(sum(fractions) - 1) < eps
    assert len(fractions) > 0
    if len(fractions) == 1:
        return [items]

    part_first, part_rest = model_selection.train_test_split(
        items, train_size=fractions[0], random_state=seed
    )
    items_split = [part_first] + shuffle_multi_split(
        part_rest, normalize(fractions[1:]), seed
    )
    assert len(items) == sum(map(len, items_split))
    return items_split


if __name__ == "__main__":
    print(HyperParams().to_string())
