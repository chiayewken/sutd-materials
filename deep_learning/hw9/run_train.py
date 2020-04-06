import copy
from pathlib import Path
from typing import Dict, List

import fire
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

from datasets import StarTrekCharGenerationDataset
from models import SequenceNet
from utils import HyperParams, Splits, get_device, Sampler


class TrainResult:
    def __init__(self, hparams: HyperParams, history: List[dict], weights: dict):
        self.hparams = hparams
        self.history = history
        self.weights = weights

    def to_dict(self) -> dict:
        s = copy.deepcopy(self.__dict__)
        s["hparams"] = s["hparams"].__dict__
        return s

    @staticmethod
    def from_dict(s: dict):
        s["hparams"] = HyperParams(**s["hparams"])
        return TrainResult(**s)

    @staticmethod
    def batch_save(results: list, path: str):
        assert all([isinstance(r, TrainResult) for r in results])
        torch.save([r.to_dict() for r in results], path)

    @staticmethod
    def batch_load(path: str):
        raw: list = torch.load(path, map_location=get_device())
        return [TrainResult.from_dict(s) for s in raw]

    def get_summary(self) -> dict:
        summary = self.hparams.__dict__
        latest: Dict[str, Dict[str, float]] = self.history[-1]
        for data_split, metrics in latest.items():
            for key, val in metrics.items():
                summary[data_split + "_" + key] = val
        return summary

    @staticmethod
    def batch_summary(results: list) -> pd.DataFrame:
        assert all([isinstance(r, TrainResult) for r in results])
        r: TrainResult
        df = pd.DataFrame([r.get_summary() for r in results])
        cols_show = [k for k in df.keys() if df[k].nunique() > 1]
        return df[cols_show]


class CharGenerationSystem:
    def __init__(
        self, hparams: HyperParams,
    ):
        self.hparams = hparams
        self.data_splits = [Splits.train, Splits.val, Splits.test]
        self.device = get_device()
        self.datasets = {s: self.get_dataset(s) for s in self.data_splits}
        self.vocab_size = len(self.datasets[Splits.train].vocab)
        self.net = SequenceNet(n_vocab=self.vocab_size, hparams=hparams,)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=hparams.lr)

    def get_dataset(self, data_split: str):
        return StarTrekCharGenerationDataset(self.hparams, data_split)

    def get_loader(self, data_split: str, bs: int) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.datasets[data_split],
            batch_size=bs,
            shuffle=(data_split == Splits.train),
        )

    def run_step(self, inputs, targets):
        outputs, states = self.net(inputs)

        outputs = outputs.reshape(-1, self.vocab_size)
        targets = targets.reshape(-1)
        loss = self.criterion(outputs, targets)

        predicts = torch.argmax(outputs.data, dim=1)
        acc = predicts.eq(targets).float().mean()
        return loss, acc

    def get_gradient_context(self, is_train: bool):
        if is_train:
            self.net.train()
            return torch.enable_grad
        else:
            self.net.eval()
            return torch.no_grad

    def run_epoch(self, data_split: str) -> Dict[str, float]:
        is_train = data_split == Splits.train
        acc_history = []
        loss_history = []
        steps_per_epoch = self.hparams.steps_per_epoch
        if data_split in {Splits.val, Splits.test}:
            steps_per_epoch = steps_per_epoch // 10

        with self.get_gradient_context(is_train)():
            while True:
                for inputs, targets in self.get_loader(data_split, self.hparams.bs):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    if is_train:
                        self.optimizer.zero_grad()

                    loss, acc = self.run_step(inputs, targets)
                    acc_history.append(acc.item())
                    loss_history.append(loss.item())

                    if is_train:
                        loss.backward()
                        self.optimizer.step()

                    if self.hparams.dev_run or len(loss_history) > steps_per_epoch:
                        acc = np.round(np.mean(acc_history), decimals=3)
                        loss = np.round(np.mean(loss_history), decimals=3)
                        return dict(loss=loss, acc=acc)

    def train(self, early_stop=True) -> TrainResult:
        self.net.to(self.device)
        best_loss = 1e9
        best_weights = {}

        history = []
        for _ in tqdm(range(self.hparams.epochs)):
            hist = {s: self.run_epoch(s) for s in self.data_splits}
            history.append(hist)
            print(hist)
            loss = hist[Splits.val]["loss"]
            if loss < best_loss:
                best_loss = loss
                best_weights = copy.deepcopy(self.net.state_dict())
            elif early_stop:
                break
            if self.hparams.dev_run and len(history) >= 3:
                break

        return TrainResult(self.hparams, history, best_weights)

    def sample(self, num: int = None, length: int = None):
        if num is None:
            num = self.hparams.bs
        if length is None:
            length = self.hparams.seq_len

        dataset: StarTrekCharGenerationDataset = self.datasets[Splits.train]
        token_start = dataset.vocab.stoi[dataset.vocab.start]
        x = torch.from_numpy(np.array([token_start] * num))
        x = x.long().reshape(num, 1)

        self.net.eval()
        sampler = dict(tcn=self.sample_tcn, lstm=self.sample_rnn, gru=self.sample_rnn)
        with torch.no_grad():
            outputs = sampler[self.hparams.model](x, length)

        for i in range(num):
            print(dataset.sequence_to_text(outputs[i, :]))

    def sample_rnn(self, x: torch.Tensor, length: int) -> torch.Tensor:
        states = None
        history = [x]
        for _ in range(length):
            logits, states = self.net(history[-1], states)
            next_logits = logits[:, -1, :]
            history.append(Sampler.sample(next_logits))
        return torch.stack(history).squeeze().transpose(0, 1)

    def sample_tcn(self, x: torch.Tensor, length: int) -> torch.Tensor:
        states = None
        history = [x]
        for _ in range(length):
            inputs = torch.cat(history[-length:], dim=-1)
            logits, states = self.net(inputs, states)
            next_logits = logits[:, -1, :]
            history.append(Sampler.temperature(next_logits))
        return torch.stack(history).squeeze().transpose(0, 1)


def main(dev_run=False, path_results="train_results.pt"):
    results: List[TrainResult] = []
    if Path(path_results).exists():
        results = TrainResult.batch_load(path_results)

    for model in ["lstm", "gru", "tcn"]:
        for n_layers in [1, 2, 3]:
            for dropout in [0.0, 0.2]:
                for n_hidden in [128, 256]:
                    hparams = HyperParams(
                        model=model,
                        n_layers=n_layers,
                        n_hidden=n_hidden,
                        dropout=dropout,
                        dev_run=dev_run,
                    )
                    if hparams not in [r.hparams for r in results]:
                        system = CharGenerationSystem(hparams)
                        results.append(system.train())

    TrainResult.batch_save(results, path_results)


if __name__ == "__main__":
    fire.Fire(main)
