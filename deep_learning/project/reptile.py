from copy import deepcopy
from typing import Dict, Tuple

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from datasets import (
    IntentEmbedBertMetaLoader,
    Splits,
    MetaBatch,
    MetaLoader,
    MetaTask,
)
from models import LinearClassifier, LinearEmbedder
from utils import (
    MathDict,
    LinearDecayLR,
    MetricsTracker,
    get_device,
    set_random_state,
    HyperParams,
    generate,
    RepeatTensorDataset,
    EarlyStopSaver,
    accuracy_score,
    cosine_distance,
)


class ReptileSGD:
    """
    Apply Reptile gradient update to weights, average gradient across meta batches
    For gradient calculation, please refer to get_gradients()
    """

    def __init__(self, net: torch.nn.Module, lr_schedule: LinearDecayLR):
        self.net = net
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
        if self.counter == 0:
            self.grads = g
        else:
            self.grads = self.grads + g
        self.counter += 1

    def step(self, i_step: int):
        grads_avg = self.grads / self.counter
        lr = self.lr_schedule.get_lr(i_step)
        weights_new = self.weights_before - (grads_avg * lr)
        self.net.load_state_dict(weights_new.state)


class MetaLearnSystem:
    """
    The Reptile meta-learning algorithm was invented by OpenAI
    https://openai.com/blog/reptile/

    System to take in meta-learning data and any kind of model
    and run Reptile meta-learning training loop
    The objective of meta-learning is not to master any single task
    but instead to obtain x1 system that can quickly adapt to x1 new task
    using x1 small number of training steps and data, like x1 human

    Reptile pseudo-code:
    Initialize initial weights, w
    For iterations:
        Randomly sample x1 task T
        Perform k steps of SGD on T, now having weights, w_after
        Update w = w - learn_rate * (w - w_after)
    """

    def __init__(self, hp: HyperParams):
        set_random_state(hp.random_seed)
        self.hp = hp
        self.device = get_device()
        self.net, self.loaders, self.opt_inner = self.configure()
        self.criterion = torch.nn.CrossEntropyLoss()
        lr_schedule = LinearDecayLR(hp.lr_outer, hp.steps_outer)
        self.opt_outer = ReptileSGD(self.net, lr_schedule)
        self.samples = {s: next(iter(loader)) for s, loader in self.loaders.items()}
        self.support = (torch.empty(0), torch.empty(0))
        self.save_path = f"{self.hp.root}/{self.hp.to_string()}.pt"
        print(dict(save_path=self.save_path))

    def configure(self) -> Tuple[torch.nn.Module, Dict[str, MetaLoader], Optimizer]:
        loader_class = dict(embed_bert=IntentEmbedBertMetaLoader)[self.hp.loader]
        loaders = {s: loader_class(self.hp, s) for s in Splits.get_all()}
        embed_size = loaders[Splits.train].embed_size
        net_class = dict(reptile=LinearClassifier, prototype=LinearEmbedder)[
            self.hp.algo
        ]
        net = net_class(num_in=embed_size, hp=self.hp)
        net = net.to(self.device)
        opt_inner_class = dict(sgd=torch.optim.SGD, adam=torch.optim.Adam)[
            self.hp.opt_inner
        ]
        opt_inner = opt_inner_class(net.parameters(), self.hp.lr_inner)
        return net, loaders, opt_inner

    def get_gradient_context(self, is_train: bool) -> torch.autograd.grad_mode:
        if is_train:
            self.net.train()
            return torch.enable_grad
        else:
            self.net.eval()
            return torch.no_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hp.algo == "reptile":
            return self.net(x)
        elif self.hp.algo == "prototype":
            distance_fn = cosine_distance
            prototypes = self.get_prototypes(*self.support)
            x = self.net(x)
            distances = distance_fn(x, prototypes)
            return distances * -1
        else:
            raise ValueError(f"Unknown meta algorithm: {self.hp.algo}")

    def get_prototypes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        embeds: torch.Tensor = self.net(x)
        prototypes = []
        for i in range(self.hp.num_ways):
            centroid = embeds[y.eq(i)].mean(dim=0)
            if y.eq(i).float().sum().item() == 0:  # label i is not present in y
                centroid = embeds.mean(dim=0)
            prototypes.append(centroid)
        return torch.stack(prototypes)

    def run_batch(
        self, batch: Tuple[torch.Tensor, torch.Tensor], is_train=True
    ) -> Dict[str, float]:
        with self.get_gradient_context(is_train)():
            inputs, targets = batch
            if is_train:
                self.opt_inner.zero_grad()
            outputs = self.forward(inputs)
            loss = self.criterion(outputs, targets)
            acc = accuracy_score(outputs, targets)
            if is_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(self, task: MetaTask) -> Dict[str, float]:
        self.support = task.train
        ds = RepeatTensorDataset(*task.train, num_repeat=self.hp.bs_inner)
        loader = DataLoader(ds, self.hp.bs_inner, shuffle=True)
        steps_per_epoch = self.hp.num_shots * self.hp.num_ways // self.hp.bs_inner
        stop_saver = EarlyStopSaver(self.net)

        for i, batch in enumerate(generate(loader, limit=self.hp.steps_inner)):
            self.run_batch(batch, is_train=True)
            if i % steps_per_epoch == 0 and self.hp.early_stop:
                metrics = self.run_batch(task.val, is_train=False)
                if stop_saver.check_stop(metrics["loss"]):
                    stop_saver.load_best()
                    break
        return self.run_batch(task.test, is_train=False)

    def loop_outer(self):
        steps = self.hp.steps_outer
        interval = steps // 10
        loader = self.loaders[Splits.train]
        tracker = MetricsTracker(prefix=Splits.train)
        stop_saver = EarlyStopSaver(self.net)

        for i, batch in enumerate(generate(loader, limit=steps, show_progress=True)):
            self.opt_outer.zero_grad()
            for task in MetaBatch(batch, self.device).get_tasks():
                metrics = self.loop_inner(task)
                tracker.store(metrics)
                self.opt_outer.store_grad()
            self.opt_outer.step(i)

            if i % interval == 0:
                metrics = tracker.get_average()
                tracker.reset()
                metrics.update(self.loop_eval(Splits.val))
                print({k: round(v, 3) for k, v in metrics.items()})
                stop_saver.check_stop(metrics["val_loss"])
            self.save()
        stop_saver.load_best()
        self.save()

    def loop_eval(self, data_split: str) -> dict:
        tracker = MetricsTracker(prefix=data_split)
        net_before = deepcopy(self.net.state_dict())
        opt_before = deepcopy(self.opt_inner.state_dict())

        def reset_state():
            self.net.load_state_dict(deepcopy(net_before))
            self.opt_inner.load_state_dict(deepcopy(opt_before))

        for task in MetaBatch(self.samples[data_split], self.device).get_tasks():
            metrics = self.loop_inner(task)
            tracker.store(metrics)
            reset_state()

        return tracker.get_average()

    def run_train(self):
        def run_eval():
            self.save()
            self.load()
            print(self.loop_eval(Splits.val))
            print(self.loop_eval(Splits.test))

        run_eval()
        self.loop_outer()
        run_eval()

    def save(self):
        state = dict(net=self.net.state_dict(), opt_inner=self.opt_inner.state_dict())
        torch.save(state, str(self.save_path))

    def load(self):
        state = torch.load(str(self.save_path), map_location=self.device)
        self.net.load_state_dict(state["net"])
        self.opt_inner.load_state_dict(state["opt_inner"])


# def run_omniglot(root: str):
#     hp = HyperParams(root=root)
#     loaders = {s: OmniglotMetaLoader(hp, s) for s in ["train", "val"]}
#     net = ConvClassifier(num_in=1, hp=hp)
#     system = MetaLearnSystem(hp)
#     system.run_train()


def get_hparams_intent(algo: str) -> HyperParams:
    return HyperParams(
        algo=algo,
        bs_inner=1,
        num_shots=5,
        early_stop=True,
        steps_inner=1000,
        steps_outer=50,
    )


def main():
    hp = get_hparams_intent(algo="reptile")
    system = MetaLearnSystem(hp)
    system.run_train()


if __name__ == "__main__":
    main()
