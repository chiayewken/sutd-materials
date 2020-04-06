import copy
from typing import Dict

import torch
import torch.utils.data
import torchmeta
import tqdm

import datasets
import models
import utils


class ReptileSGD:
    """
    Apply Reptile gradient update to weights, average gradient across meta batches
    The Reptile gradient is quite simple, please refer to "get_gradients"
    The model weights are using SGD in "step"
    """

    def __init__(self, net: torch.nn.Module, lr_schedule, num_accum):
        self.net = net
        self.num_accum = num_accum
        self.lr_schedule: utils.LinearDecayLR = lr_schedule

        self.grads = utils.MathDict()
        self.weights_before = utils.MathDict()
        self.counter = 0
        self.zero_grad()

    def zero_grad(self):
        self.grads = utils.MathDict()
        self.weights_before = utils.MathDict(self.net.state_dict())
        self.counter = 0

    def get_gradients(self):
        weights_after = utils.MathDict(self.net.state_dict())
        return self.weights_before - weights_after

    def store_grad(self):
        g = self.get_gradients()
        assert self.counter < self.num_accum
        if self.counter == 0:
            self.grads = g
        else:
            self.grads = self.grads + g
        self.counter += 1

    def step(self, i_step):
        assert self.counter == self.num_accum
        grads_avg = self.grads / self.num_accum
        lr = self.lr_schedule.get_lr(i_step)
        weights_new = self.weights_before - (grads_avg * lr)
        self.net.load_state_dict(weights_new.state)


class HyperParams(datasets.MetaDataParams):
    def __init__(
        self,
        lr_inner=1e-3,
        lr_outer=1.0,
        lr_val=1e-3,
        steps_inner=5,
        steps_outer=1000,
        steps_val=50,
        bs_inner=10,
        bs_val=5,
        **kwargs
    ):
        self.lr_inner = lr_inner
        self.lr_outer = lr_outer
        self.lr_val = lr_val
        self.steps_inner = steps_inner
        self.steps_outer = steps_outer
        self.steps_val = steps_val
        self.bs_inner = bs_inner
        self.bs_val = bs_val
        super().__init__(**kwargs)


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

        self.device = utils.get_device()
        self.rng = utils.set_random_state()
        self.net = net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt_inner = torch.optim.SGD(self.net.parameters(), lr=hparams.lr_inner)
        lr_schedule = utils.LinearDecayLR(hparams.lr_outer, hparams.steps_outer)
        self.opt_outer = ReptileSGD(self.net, lr_schedule, num_accum=hparams.bs)
        self.batch_val = next(iter(self.loaders[datasets.Splits.val]))

    def run_batch(self, batch, do_train=True):
        if do_train:
            self.net.train()
            context_grad = torch.enable_grad
        else:
            self.net.eval()
            context_grad = torch.no_grad

        with context_grad():
            inputs, targets = batch
            if do_train:
                self.opt_inner.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            acc = utils.acc_score(outputs, targets)
            if do_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(self, task, bs, steps):
        x_train, y_train, x_test, y_test = task
        ds = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(ds, bs, shuffle=True)

        i = 0
        while i < steps:
            for batch in loader:
                self.run_batch(batch, do_train=True)
                i += 1

        metrics = self.run_batch((x_test, y_test), do_train=False)
        return metrics

    def loop_outer(self):
        steps = self.hparams.steps_outer
        interval = steps // 100
        loader = self.loaders[datasets.Splits.train]
        tracker = utils.MetricsTracker(prefix=datasets.Splits.train)

        with tqdm.tqdm(loader, total=steps) as pbar:
            for i, batch in enumerate(pbar):
                if i > steps:
                    break

                self.opt_outer.zero_grad()
                for task in datasets.MetaBatch(batch, self.device).get_tasks():
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
                    pbar.set_postfix(metrics)

    def loop_val(self):
        tracker = utils.MetricsTracker(prefix=datasets.Splits.val)
        net_before = copy.deepcopy(self.net.state_dict())
        opt_before = copy.deepcopy(self.opt_inner.state_dict())

        def reset_state():
            self.net.load_state_dict(net_before)
            self.opt_inner.load_state_dict(opt_before)

        for task in datasets.MetaBatch(self.batch_val, self.device).get_tasks():
            reset_state()
            metrics = self.loop_inner(task, self.hparams.bs_val, self.hparams.steps_val)
            tracker.store(metrics)

        reset_state()
        return tracker.get_average()

    def run_train(self):
        self.loop_outer()


def run_omniglot(root):
    hparams = HyperParams(root=root)
    loaders = {s: datasets.OmniglotMetaLoader(hparams, s) for s in ["train", "val"]}
    net = models.ConvClassifier(size_in=1, size_out=hparams.num_ways)
    system = ReptileSystem(hparams, loaders, net)
    system.run_train()


def run_intent(root):
    hparams = HyperParams(root=root, steps_outer=500, steps_inner=50, bs_inner=5)
    loaders = {s: datasets.IntentMetaLoader(hparams, s) for s in ["train", "val"]}
    net = models.LinearClassifier(
        size_in=loaders["train"].embedder.size_embed, size_out=hparams.num_ways
    )
    system = ReptileSystem(hparams, loaders, net)
    system.run_train()


def main(root="temp"):
    run_intent(root)


if __name__ == "__main__":
    main()
