from copy import deepcopy
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from datasets import (
    OmniglotMetaLoader,
    IntentEmbedBertMetaLoader,
    Splits,
    MetaBatch,
    MetaLoader,
    MetaTask,
    IntentEmbedWordMetaLoader,
    IntentEmbedWordMeanMetaLoader,
)
from models import ConvClassifier, LinearClassifier, LSTMClassifier
from utils import (
    MathDict,
    LinearDecayLR,
    MetricsTracker,
    get_device,
    set_random_state,
    HyperParams,
    generate,
    RepeatTensorDataset,
    EarlyStopCallback,
)


class ReptileSGD:
    """
    Apply Reptile gradient update to weights, average gradient across meta batches
    The Reptile gradient is quite simple, please refer to "get_gradients"
    The model weights are using SGD in "step"
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


class ReptileSystem:
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

    def __init__(
        self,
        hp: HyperParams,
        loaders: Dict[str, MetaLoader],
        net: torch.nn.Module,
        use_gpu=True,
    ):
        self.hp = hp
        self.loaders = loaders

        self.device = get_device(use_gpu)
        self.rng = set_random_state()
        self.net = net.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.opt_inner = torch.optim.SGD(self.net.parameters(), lr=hp.lr_inner)
        lr_schedule = LinearDecayLR(hp.lr_outer, hp.steps_outer)
        self.opt_outer = ReptileSGD(self.net, lr_schedule)
        self.batch_val = next(iter(self.loaders[Splits.val]))

    def get_gradient_context(self, is_train: bool) -> torch.autograd.grad_mode:
        if is_train:
            self.net.train()
            return torch.enable_grad
        else:
            self.net.eval()
            return torch.no_grad

    @staticmethod
    def get_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds: torch.Tensor
        if logits.shape[-1] == 1:
            preds = logits.sigmoid().round()
        else:
            preds = torch.argmax(logits, dim=-1)
        return (preds == targets).float().mean()

    def run_batch(
        self, batch: Tuple[torch.Tensor, torch.LongTensor], is_train=True
    ) -> Dict[str, float]:
        with self.get_gradient_context(is_train)():
            inputs, targets = batch
            if is_train:
                self.opt_inner.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            acc = self.get_accuracy(outputs, targets)
            if is_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(self, task: MetaTask) -> Dict[str, float]:
        ds = RepeatTensorDataset(*task.train, num_repeat=self.hp.bs_inner)
        loader = DataLoader(ds, self.hp.bs_inner, shuffle=True)
        steps_per_epoch = self.hp.num_shots * self.hp.num_ways // self.hp.bs_inner
        stopper = EarlyStopCallback(self.net)

        for i, batch in enumerate(generate(loader, limit=self.hp.steps_inner)):
            self.run_batch(batch, is_train=True)
            if i % steps_per_epoch == 0 and self.hp.early_stop:
                metrics = self.run_batch(task.val, is_train=False)
                if stopper.check_stop(metrics["loss"]):
                    stopper.load_best()
                    break
        return self.run_batch(task.test, is_train=False)

    def loop_outer(self):
        steps = self.hp.steps_outer
        interval = steps // 10
        loader = self.loaders[Splits.train]
        tracker = MetricsTracker(prefix=Splits.train)

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
                metrics.update(self.loop_val())
                print({k: round(v, 3) for k, v in metrics.items()})

    def loop_val(self) -> dict:
        tracker = MetricsTracker(prefix=Splits.val)
        net_before = deepcopy(self.net.state_dict())
        opt_before = deepcopy(self.opt_inner.state_dict())

        def reset_state():
            self.net.load_state_dict(deepcopy(net_before))
            self.opt_inner.load_state_dict(deepcopy(opt_before))

        for task in MetaBatch(self.batch_val, self.device).get_tasks():
            metrics = self.loop_inner(task)
            tracker.store(metrics)
            reset_state()

        return tracker.get_average()

    def run_train(self):
        self.loop_outer()


# Query examples are the test samples while Support examples are the training examples
class PrototypeSystem(ReptileSystem):
    def __init__(
        self,
        hparams: HyperParams,
        loaders: Dict[str, MetaLoader],
        net: torch.nn.Module,
    ):
        super().__init__(hparams, loaders, net)

    @staticmethod
    def euclidean_dist(query: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        # query: N x D
        # prototype: M x D
        n = query.size(0)
        m = prototype.size(0)

        d = query.size(1)

        assert d == prototype.size(1)

        query = query.unsqueeze(1).expand(n, m, d)
        prototype = prototype.unsqueeze(0).expand(n, m, d)

        return torch.pow(query - prototype, 2).sum(2)

    def run_batch(
        self,
        batch: Tuple[torch.Tensor, torch.LongTensor],
        prototypes: torch.Tensor,
        is_train=True,
    ) -> Dict[str, float]:
        inputs, targets = batch
        num_class = len(prototypes)
        num_queries = inputs.shape[0]
        with self.get_gradient_context(is_train)():
            if is_train:
                self.opt_inner.zero_grad()
            outputs = self.net(inputs)
            # outputs shape is batch_size, 64 prototypes shape is 64

            # find euclid distance between x_test and prototypes
            dists = self.euclidean_dist(outputs, prototypes)
            # do log softmax
            log_p_y_prob = F.log_softmax(-dists, dim=1)

            # get the argmax of each query
            _, y_hat = log_p_y_prob.max(1)

            # get -ve log softmax
            log_p_y = -log_p_y_prob
            # gather loss values of each query with reference to true labels in target tensor
            loss = []
            for i in range(len(targets)):
                loss += [log_p_y[i][targets[i]]]
            # convert loss array to loss tensor and get avg loss
            loss = torch.stack(loss).sum() / (num_class * num_queries)

            # compare to true labels to get accuracy
            acc = torch.eq(y_hat, targets).float().mean()

            if is_train:
                loss.backward()
                self.opt_inner.step()
        return dict(loss=loss.item(), acc=acc.item())

    def loop_inner(self, task: MetaTask) -> Dict[str, float]:
        ds = RepeatTensorDataset(*task.train, num_repeat=self.hp.bs_inner)
        loader = DataLoader(ds, self.hp.bs_inner, shuffle=True)
        steps_per_epoch = self.hp.num_shots * self.hp.num_ways // self.hp.bs_inner
        stopper = EarlyStopCallback(self.net)

        for i, batch in enumerate(generate(loader, limit=self.hp.steps_inner)):
            self.run_batch(batch, is_train=True)
            if i % steps_per_epoch == 0 and self.hp.early_stop:
                metrics = self.run_batch(task.val, is_train=False)
                if stopper.check_stop(metrics["loss"]):
                    stopper.load_best()
                    break
        return self.run_batch(task.test, is_train=False)

    def loop_inner(self, task: MetaTask) -> Dict[str, float]:
        ds = RepeatTensorDataset(*task.train, num_repeat=self.hp.bs_inner)
        loader = DataLoader(ds, self.hp.bs_inner, shuffle=True)
        steps_per_epoch = self.hp.num_shots * self.hp.num_ways // self.hp.bs_inner
        stopper = EarlyStopCallback(self.net)

        proto = self.get_prototypes(*task.train)

        # calculate mean support for each embedding
        for batch in generate(loader, limit=steps):
            self.run_batch(batch, prototypes, is_train=True)

        # run the different training method
        return self.run_batch((x_test, y_test), prototypes, is_train=False)

    def get_prototypes(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            embeds = self.net(x)
            # embeds train is of shape 25,64
            # thus prototypes of each class should be of shape 1,64 each.
            prototypes = torch.zeros(self.hp.num_ways, x.shape[-1])

            for class_index in range(self.hp.num_ways):
                mask = torch.eq(y, class_index)
                if mask.sum() == 0:
                    pass
                else:
                    cluster = embeds[mask]
                    prototypes[class_index] = torch.mean(cluster, dim=0)

            return prototypes.to(self.device)


def run_intent(root: str):
    hp = HyperParams(
        root=root,
        bs_inner=1,
        num_shots=5,
        early_stop=True,
        # steps_inner=50,
        steps_inner=1000,
        steps_outer=50,
        # steps_outer=500,
    )
    # loader_class = IntentEmbedBertMetaLoader
    loader_class = IntentEmbedWordMeanMetaLoader
    loaders = {s: loader_class(hp, s) for s in Splits.get_all()}
    load = loaders[Splits.train]
    net = LinearClassifier(num_in=load.embed_size, hp=hp)
    # net = LSTMClassifier(num_in=load.embed_size, hp=hp)
    system = ReptileSystem(hp, loaders, net)
    system.run_train()


def main(root="data"):
    run_intent(root)


if __name__ == "__main__":
    main()
