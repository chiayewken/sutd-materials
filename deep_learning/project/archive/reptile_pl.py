import argparse
import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
import fire


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x_all, gen_task_fn, epochs_inner, epochs_outer, batch_size, rng):
        self.batch_size = batch_size
        self.epochs_inner = epochs_inner
        self.epochs_outer = epochs_outer
        self.num_samples_orig = x_all.shape[0]
        self.batches_per_task = (
            self.epochs_inner * self.num_samples_orig // self.batch_size
        )
        self.samples_per_task = self.batches_per_task * self.batch_size
        print(dict(MyDataset=self.__dict__))

        self.x_all = torch.from_numpy(x_all).float()
        self.samples = []
        self.y_tasks = []
        self.task_fns = [gen_task_fn() for _ in range(epochs_outer)]

        for e in range(epochs_outer):
            task_fn = gen_task_fn()
            y_task = task_fn(x_all)
            y_task = torch.from_numpy(y_task).float()
            assert y_task.shape[0] == self.num_samples_orig
            self.y_tasks.append(y_task)
            for i in rng.choice(self.num_samples_orig, size=self.samples_per_task):
                self.samples.append((e, i))

    def __getitem__(self, item):
        idx_epoch_outer, idx_sample = self.samples[item]
        x = self.x_all[idx_sample]
        y = self.y_tasks[idx_epoch_outer][idx_sample]
        return x, y

    def __len__(self):
        return self.epochs_outer * self.samples_per_task


class MyModel(pl.LightningModule):
    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        seed = 0
        plot = True
        innerstepsize = 0.02  # stepsize in inner SGD
        innerepochs = 1  # number of epochs of each inner SGD
        outerstepsize0 = 0.1  # stepsize of outer optimization, i.e., meta-optimization
        niterations = (
            30000
        )  # number of outer updates; each iteration we sample one task and update on it

        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # Define task distribution
        x_all = np.linspace(-5, 5, 50)[:, None]  # All of the x points
        ntrain = 10  # Size of training minibatches

        def gen_task():
            # Generate classification problem
            phase = rng.uniform(low=0, high=2 * np.pi)
            ampl = rng.uniform(0.1, 5)
            f_randomsine = lambda x: np.sin(x + phase) * ampl
            return f_randomsine

        # Define model. Reptile paper uses ReLU, but Tanh gives slightly better results
        num_hidden = 64
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(num_hidden, num_hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(num_hidden, 1),
        )
        self.ds_train = MyDataset(
            x_all, gen_task, innerepochs, niterations, ntrain, rng
        )
        self.criterion = torch.nn.MSELoss()
        self.lr_inner = innerstepsize
        self.weights_before = copy.deepcopy(self.state_dict())
        self.lr_outer = outerstepsize0
        self.epochs_outer = niterations

    def forward(self, x):
        return self.net(x)

    def train_dataloader(self):
        ds = self.ds_train
        return torch.utils.data.DataLoader(
            ds, ds.batch_size, shuffle=False, num_workers=2
        )

    def training_step(self, batch, idx_batch):
        x, y = batch
        preds = self.net(x)
        loss = self.criterion(preds, y)
        self.outer_update(idx_batch)
        log = dict(train_loss=loss)
        return dict(loss=loss, idx_batch=idx_batch, progress_bar=log, log=log)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_inner)

    def outer_update(self, idx_batch):
        with torch.no_grad():
            bpt = self.ds_train.batches_per_task
            if idx_batch % bpt == 0:
                idx_epoch_outer = idx_batch // bpt
                w_after = self.state_dict()
                w_before = self.weights_before
                lr = self.lr_outer_schedule(idx_epoch_outer)
                weights = {}
                for k, wb in w_before.items():
                    weights[k] = wb + (w_after[k] - wb) * lr
                self.load_state_dict(weights)

    def lr_outer_schedule(self, idx_epoch_outer):
        # Linear decay
        return self.lr_outer * (1 - idx_epoch_outer / self.epochs_outer)


def check_gpu():
    gpus = 1 if torch.cuda.is_available() else None
    try:
        import apex

        apex.amp.register_float_function(torch, "sigmoid")
        use_amp = True  # Mixed precision training, slightly faster
    except ImportError:
        use_amp = False
    print(dict(gpus=gpus, use_amp=use_amp))
    return gpus, use_amp


def main(fast_dev_run=False, default_save_path="temp"):
    model = MyModel(argparse.Namespace(**locals()))
    gpus, use_amp = check_gpu()
    trainer = pl.Trainer(
        gpus=gpus,
        use_amp=use_amp,
        fast_dev_run=fast_dev_run,
        max_epochs=100,
        default_save_path=default_save_path,
        logger=False,
    )
    trainer.fit(model)


if __name__ == "__main__":
    fire.Fire(main)
