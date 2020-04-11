import operator
import time
from collections import Iterable
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import model_selection
from tqdm import tqdm


class HyperParams:
    def __init__(
        self,
        root="temp",
        num_ways=5,
        num_shots=5,
        num_shots_test=5,
        lr_inner=1e-3,
        lr_outer=1.0,
        steps_inner=5,
        steps_outer=1000,
        steps_val=50,
        bs_inner=10,
        bs_outer=5,
        num_hidden=64,
        num_layers=3,
    ):
        # Data
        self.root = root
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_shots_test = num_shots_test

        # Training
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.steps_inner = steps_inner
        self.steps_outer = steps_outer
        self.steps_val = steps_val
        self.bs_inner = bs_inner
        self.bs_outer = bs_outer

        # Model
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        print(vars(self))


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


def set_random_state(seed=0):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    return rng


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(dict(device=device))
    return device


class Timer:
    def __init__(self, name, decimals=1):
        self.name = name
        self.decimals = decimals

    def __enter__(self):
        print("Start timer:", self.name)
        self.start = time.time()

    def __exit__(self, type, value, traceback):
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
