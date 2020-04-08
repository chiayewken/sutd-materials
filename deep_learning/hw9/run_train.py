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
    def __init__(
        self,
        hparams: HyperParams,
        history: List[dict],
        weights: Dict[str, torch.Tensor],
    ):
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

    def get_summary(self) -> dict:
        summary = copy.deepcopy(self.hparams.__dict__)
        latest: Dict[str, Dict[str, float]] = self.history[-1]
        for data_split, metrics in latest.items():
            for key, val in metrics.items():
                summary[data_split + "_" + key] = val
        return summary


class ResultsManager:
    def __init__(self, path: str):
        self.path = Path(path)
        self.results: List[TrainResult] = []
        if self.path.exists():
            self.load()

    def save(self):
        torch.save([r.to_dict() for r in self.results], str(self.path))

    def load(self):
        print(dict(loading=self.path))
        raw: list = torch.load(str(self.path), map_location=get_device())
        self.results = [TrainResult.from_dict(s) for s in raw]

    def get_summary(self) -> pd.DataFrame:
        df = pd.DataFrame([r.get_summary() for r in self.results])
        cols_changed = [k for k in df.keys() if df[k].nunique() > 1]
        if cols_changed:
            df = df[cols_changed].sort_values("val_loss", ascending=False)
        return df

    def get_best(self):
        sort = sorted(self.results, key=lambda r: r.history[-1][Splits.val]["loss"])
        best = sort[0]
        print(dict(best=best.hparams))
        return best

    def check_hparams_exist(self, hparams: HyperParams) -> bool:
        r: TrainResult
        return hparams in [r.hparams for r in self.results]

    def add(self, r: TrainResult):
        assert not self.check_hparams_exist(r.hparams)
        self.results.append(r)


class CharGenerationSystem:
    def __init__(
        self, hparams: HyperParams,
    ):
        self.hparams = hparams
        self.data_splits = [Splits.train, Splits.val, Splits.test]
        self.device = get_device(verbose=self.hparams.verbose)
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

    def run_train(self, early_stop=True) -> TrainResult:
        self.net.to(self.device)
        best_loss = 1e9
        best_weights = {}

        history = []
        for _ in tqdm(range(self.hparams.epochs), disable=(not self.hparams.verbose)):
            hist = {s: self.run_epoch(s) for s in self.data_splits}
            history.append(hist)
            if self.hparams.verbose:
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

    @classmethod
    def load(cls, hparams: HyperParams, weights: Dict[str, torch.Tensor]):
        system = cls(hparams)
        system.net.load_state_dict(weights)
        return system

    def sample(self, num: int = None, length: int = None) -> List[str]:
        if num is None:
            num = self.hparams.bs
        if length is None:
            length = self.hparams.seq_len

        dataset: StarTrekCharGenerationDataset = self.datasets[Splits.train]
        token_start = dataset.vocab.stoi[dataset.vocab.start]
        x = torch.from_numpy(np.array([token_start] * num)).long()
        x = x.reshape(num, 1)

        self.net.eval()
        with torch.no_grad():
            states = None
            history = [x]
            for _ in tqdm(range(length)):
                n_look_back = length if self.hparams.model == "tcn" else 1
                inputs = torch.cat(history[-n_look_back:], dim=-1)
                logits, states = self.net(inputs, states)
                next_logits = logits[:, -1, :]
                history.append(Sampler.temperature(next_logits))
            history = history[1:]  # Omit start tokens
            outputs = torch.stack(history).squeeze().transpose(0, 1)

        return [dataset.sequence_to_text(outputs[i, :]) for i in range(num)]


def enumerate_grid(grid: Dict[str, list]) -> List[dict]:
    dicts: List[dict] = [{}]
    for key, values in grid.items():
        temp = []
        for d in dicts:
            for val in values:
                d = copy.deepcopy(d)
                d[key] = val
                temp.append(d)
        dicts = temp
    return dicts


def search_hparams(path, dev_run) -> HyperParams:
    manager = ResultsManager(path)
    grid = dict(
        model=["lstm", "gru", "tcn"],
        n_layers=[1, 2, 3],
        dropout=[0.0, 0.2],
        n_hidden=[128, 256],
        bs=[16, 32, 128, 256, 512],
        lr=[1e-2, 1e-3, 1e-4],
    )
    for kwargs in tqdm(enumerate_grid(grid)):
        hparams = HyperParams(epochs=1, verbose=False, dev_run=dev_run, **kwargs)
        if not manager.check_hparams_exist(hparams):
            result = CharGenerationSystem(hparams).run_train()
            result.weights = {}
            manager.add(result)
    manager.save()
    print(manager.get_summary())
    return manager.get_best().hparams


def main(
    path_results_search="results_search.pt",
    path_results_train="results_train.pt",
    dev_run=False,
):
    hparams = search_hparams(path_results_search, dev_run)
    default = HyperParams()
    hparams.verbose = default.verbose
    hparams.epochs = default.epochs

    manager = ResultsManager(path_results_train)
    if not manager.check_hparams_exist(hparams):
        manager.add(CharGenerationSystem(hparams).run_train())
        manager.save()


if __name__ == "__main__":
    fire.Fire(main)
