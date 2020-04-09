from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torchmeta
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from datasets import (
    OmniglotMetaLoader,
    IntentMetaLoader,
    Splits,
    MetaBatch,
    MetaDataParams,
)
from models import ConvClassifier, LinearClassifier
from utils import (
    MathDict,
    LinearDecayLR,
    MetricsTracker,
    get_device,
    set_random_state,
    acc_score,
)


class ReptileSGD:
    """
    Apply Reptile gradient update to weights, average gradient across meta batches
    The Reptile gradient is quite simple, please refer to "get_gradients"
    The model weights are using SGD in "step"
    """

    def __init__(
        self, net: torch.nn.Module, lr_schedule: LinearDecayLR, num_accum: int
    ):
        self.net = net
        self.num_accum = num_accum
        self.lr_schedule = lr_schedule

        self.grads = MathDict()
        self.weights_before = MathDict()
        self.counter = 0
        self.zero_grad()

    def zero_grad(self):
        self.grads = MathDict()
        self.weights_before = MathDict(self.net.state_dict())
        self.counter = 0

    def get_gradients(self) -> MathDict:
        weights_after = MathDict(self.net.state_dict())
        return self.weights_before - weights_after

    def store_grad(self):
        g = self.get_gradients()
        assert self.counter < self.num_accum
        if self.counter == 0:
            self.grads = g
        else:
            self.grads = self.grads + g
        self.counter += 1

    def step(self, i_step: int):
        assert self.counter == self.num_accum
        grads_avg = self.grads / self.num_accum
        lr = self.lr_schedule.get_lr(i_step)
        weights_new = self.weights_before - (grads_avg * lr)
        self.net.load_state_dict(weights_new.state)


class HyperParams(MetaDataParams):
    def __init__(
        self,
        lr_inner=1e-3,
        lr_outer=1.0,
        steps_inner=5,
        steps_outer=1000,
        steps_val=50,
        bs_inner=10,
        bs_outer=5,
        **kwargs
    ):
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.steps_inner = steps_inner
        self.steps_outer = steps_outer
        self.steps_val = steps_val
        self.bs_inner = bs_inner
        self.bs_outer = bs_outer
        super().__init__(bs=bs_outer, **kwargs)


class ReptileSystem:
    """
    The Reptile meta-learning algorithm was invented by OpenAI
    https://openai.com/blog/reptile/

    System to take in meta-learning data and any kind of model
    and run Reptile meta-learning training loop
    The objective of meta-learning is not to master any single task
    but instead to obtain a system that can quickly adapt to a new task
    using a small number of training steps and data, like a human

    Reptile pseudo-code:
    Initialize initial weights, w
    For iterations:
        Randomly sample a task T
        Perform k steps of SGD on T, now having weights, w_after
        Update w = w - learn_rate * (w - w_after)
    """

    def __init__(
        self,
        hparams: HyperParams,
        loaders: Dict[str, torchmeta.utils.data.BatchMetaDataLoader],
        net: torch.nn.Module,
    ):
        self.hparams = hparams
        self.loaders = loaders

        self.device = get_device()
        self.rng = set_random_state()
        self.net = net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt_inner = torch.optim.SGD(self.net.parameters(), lr=hparams.lr_inner)
        lr_schedule = LinearDecayLR(hparams.lr_outer, hparams.steps_outer)
        self.opt_outer = ReptileSGD(self.net, lr_schedule, num_accum=hparams.bs)
        self.batch_val = next(iter(self.loaders[Splits.val]))

    def get_gradient_context(self, is_train: bool) -> torch.autograd.grad_mode:
        if is_train:
            self.net.train()
            return torch.enable_grad
        else:
            self.net.eval()
            return torch.no_grad

    def run_batch(
        self, batch: Tuple[torch.Tensor, torch.LongTensor], is_train=True
    ) -> Dict[str, float]:
        with self.get_gradient_context(is_train)():
            inputs, targets = batch
            if is_train:
                self.opt_inner.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            acc = acc_score(outputs, targets)
            if is_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(
        self, task: List[torch.Tensor], bs: int, steps: int,
    ) -> Dict[str, float]:
        x_train, y_train, x_test, y_test = task
        ds = TensorDataset(x_train, y_train)
        loader = DataLoader(ds, bs, shuffle=True)

        i = 0
        while i < steps:
            for batch in loader:
                self.run_batch(batch, is_train=True)
                i += 1

        return self.run_batch((x_test, y_test), is_train=False)

    def loop_outer(self):
        steps = self.hparams.steps_outer
        interval = steps // 10
        loader = self.loaders[Splits.train]
        tracker = MetricsTracker(prefix=Splits.train)

        for i, batch in enumerate(tqdm(loader, total=steps)):
            if i > steps:
                break

            self.opt_outer.zero_grad()
            for task in MetaBatch(batch, self.device).get_tasks():
                metrics = self.loop_inner(
                    task, self.hparams.bs_inner, self.hparams.steps_inner
                )
                tracker.store(metrics)
                self.opt_outer.store_grad()
            self.opt_outer.step(i)

            if i % interval == 0:
                metrics = tracker.get_average()
                tracker.reset()
                metrics.update(self.loop_val())
                print({k: round(v, 3) for k, v in metrics.items()})

    def loop_val(self) -> dict:
        tracker = MetricsTracker(prefix=Splits.val)
        net_before = deepcopy(self.net.state_dict())
        opt_before = deepcopy(self.opt_inner.state_dict())

        def reset_state():
            self.net.load_state_dict(net_before)
            self.opt_inner.load_state_dict(opt_before)

        for task in MetaBatch(self.batch_val, self.device).get_tasks():
            reset_state()
            metrics = self.loop_inner(
                task, self.hparams.bs_inner, self.hparams.steps_val
            )
            tracker.store(metrics)

        reset_state()
        return tracker.get_average()

    def run_train(self):
        self.loop_outer()


def run_omniglot(root: str):
    hparams = HyperParams(root=root)
    loaders = {s: OmniglotMetaLoader(hparams, s) for s in ["train", "val"]}
    net = ConvClassifier(num_in=1, num_out=hparams.num_ways)
    system = ReptileSystem(hparams, loaders, net)
    system.run_train()


def run_intent(root: str):
    hparams = HyperParams(root=root, steps_outer=500, steps_inner=50, bs_inner=10)
    loaders = {s: IntentMetaLoader(hparams, s) for s in ["train", "val"]}
    net = LinearClassifier(
        num_in=loaders[Splits.train].embedder.size_embed, num_out=hparams.num_ways
    )
    system = ReptileSystem(hparams, loaders, net)
    system.run_train()


def main(root="temp"):
    run_intent(root)


if __name__ == "__main__":
    main()
