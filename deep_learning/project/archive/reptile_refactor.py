import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float()


def set_random_state(seed=0):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    return rng


def plot(x: torch.Tensor, y: torch.Tensor, label, color, marker=""):
    plt.plot(x.numpy(), y.numpy(), marker, label=label, color=color)


class ReptileSineSystem:
    def __init__(
        self,
        lr_inner=0.02,  # stepsize in inner SGD
        epochs_inner=1,  # number of epochs of each inner SGD
        lr_outer_init=0.1,  # stepsize of outer optimization, i.e., meta-optimization
        epochs_outer=30000,  # number of outer updates; each iteration we sample one task and update on it
        batch_size=10,  # Size of training minibatches
    ):
        super().__init__()
        self.batch_size = batch_size
        self.epochs_outer = epochs_outer
        self.epochs_inner = epochs_inner

        self.rng = set_random_state()
        self.net = self.get_net()
        self.criterion = torch.nn.MSELoss()
        self.x_all = self.get_task_distribution()
        self.task_fn_test = self.generate_task_fn()
        self.x_test = self.x_all[self.rng.choice(self.x_all.shape[0], size=batch_size)]
        self.opt_inner = torch.optim.SGD(self.net.parameters(), lr=lr_inner)
        self.lr_outer_schedule_fn = lambda i: lr_outer_init * (
            1 - i / self.epochs_outer
        )

    @staticmethod
    def get_net(num_hidden=64):
        # Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
        return torch.nn.Sequential(
            torch.nn.Linear(1, num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(num_hidden, 1),
        )

    @staticmethod
    def get_task_distribution():
        x = np.linspace(-5, 5, 50)
        x = to_tensor(x)
        x = x.view(-1, 1)
        return x

    def generate_task_fn(self):
        # Generate function to produce regression sine wave from inputs
        phase = self.rng.uniform(low=0, high=2 * np.pi)
        ampl = self.rng.uniform(0.1, 5)
        # fn_random_sine = lambda x: np.sin(x + phase) * ampl
        fn_random_sine = lambda x: torch.sin(x + phase) * ampl
        return fn_random_sine

    def train_on_batch(self, x, y):
        self.net.train()
        self.opt_inner.zero_grad()
        preds = self.net(x)
        loss = self.criterion(preds, y)
        loss.backward()
        self.opt_inner.step()

    def predict(self, x):
        self.net.eval()
        with torch.no_grad():
            return self.net(x)

    def inner_train_loop(self, y_task: torch.Tensor):
        num_samples = self.x_all.shape[0]
        assert num_samples == y_task.shape[0]
        idxs = self.rng.permutation(num_samples)
        for _ in range(self.epochs_inner):
            for start in range(0, num_samples, self.batch_size):
                batch = idxs[start : start + self.batch_size]
                self.train_on_batch(self.x_all[batch], y_task[batch])

    def eval_plot(self, epoch, fast_adapt_batches=32):
        # Periodically plot the results on a particular task and minibatch
        plt.cla()
        weights_before = copy.deepcopy(self.net.state_dict())
        y_test, y_all = map(self.task_fn_test, [self.x_test, self.x_all])
        plot(self.x_all, self.predict(self.x_all), "pred no adapt", color=(0, 0, 1))

        for i in range(fast_adapt_batches):
            self.train_on_batch(self.x_test, y_test)
            if (i + 1) % 8 == 0:
                frac = (i + 1) / fast_adapt_batches
                color = (frac, 0, 1 - frac)
                plot(self.x_all, self.predict(self.x_all), f"pred after {i + 1}", color)

        plot(self.x_all, y_all, "true", color=(0, 1, 0))
        loss = self.criterion(self.predict(self.x_all), y_all).mean().numpy()
        plot(self.x_test, y_test, "train", color="k", marker="x")
        plt.ylim(-4, 4)
        plt.legend(loc="lower right")
        plt.pause(0.01)
        self.net.load_state_dict(weights_before)  # restore from snapshot
        print(dict(epoch_outer=epoch + 1, loss=loss.round(3)))
        # would be better to average loss over a set of examples, but this is optimized for brevity

    def outer_update(self, weights_before, epoch):
        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        weights = self.net.state_dict()
        lr = self.lr_outer_schedule_fn(epoch)
        for k, wb in weights_before.items():
            weights[k] = wb - lr * (wb - weights[k])
        self.net.load_state_dict(weights)

    def train(self):
        for e in tqdm.tqdm(range(self.epochs_outer)):
            task_fn = self.generate_task_fn()
            y_task = task_fn(self.x_all)
            weights_before = copy.deepcopy(self.net.state_dict())
            self.inner_train_loop(y_task)
            self.outer_update(weights_before, e)
            if e == 0 or (e + 1) % 1000 == 0:
                self.eval_plot(e)


def main():
    system = ReptileSineSystem()
    system.train()


if __name__ == "__main__":
    main()
