import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.utils.data
from torchvision.datasets.utils import download_url
from tqdm import tqdm

from utils import HyperParams, Splits, Vocab, shuffle_multi_split


class StarTrekCharGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, hparams: HyperParams, data_split: str, sep_line="\n"):
        assert Splits.check_valid(data_split)
        self.hparams = hparams
        self.root = Path(self.hparams.root)
        self.data_split = data_split
        self.sep_line = sep_line

        self.path_data = self.download()
        self.lines = self.preprocess_data()
        self.vocab = Vocab(list(self.sep_line.join(self.lines)))
        self.text = self.train_val_test_split()
        self.tensor = self.get_sequences()
        if self.hparams.verbose:
            self.show_samples()
            print(dict(vocab_size=len(self.vocab)))

    def download(self) -> List[str]:
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/star_trek_transcripts_all_episodes.csv"
        path = self.root / Path(url).name
        if not path.exists():
            download_url(url, str(self.root), filename=path.name)
        assert path.exists()
        return path

    def preprocess_data(self) -> List[str]:
        with open(str(self.path_data)) as f:
            return [line.strip().strip(",") for line in f if "NEXTEPISODE" not in line]

    def train_val_test_split(self, fractions=(0.8, 0.1, 0.1)) -> str:
        indices_all = list(range(len(self.lines)))
        indices_split = shuffle_multi_split(indices_all, fractions)
        indices = indices_split[
            [Splits.train, Splits.val, Splits.test].index(self.data_split)
        ]
        lines = [self.lines[i] for i in indices]
        text = self.sep_line.join(lines)
        if self.hparams.verbose:
            print(dict(lines=len(lines), text=len(text)))
        return text

    def get_sequences(self) -> torch.Tensor:
        path_cache = self.root / f"cache_tensor_{self.data_split}.pt"
        token_start = self.vocab.stoi[self.vocab.start]

        if not path_cache.exists():
            encoded = self.vocab.encode(list(self.text))
            sequences = []
            for i in tqdm(range(len(encoded) - self.hparams.seq_len)):
                sequences.append([token_start] + encoded[i : i + self.hparams.seq_len])
            tensor = torch.from_numpy(np.array(sequences)).type(torch.long)
            torch.save(tensor, str(path_cache))

        tensor = torch.load(str(path_cache))
        if self.hparams.verbose:
            print(dict(tensor=tensor.shape))
        return tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, i):
        sequence = self.tensor[i, :]
        return sequence[:-1], sequence[1:]

    def sequence_to_text(self, sequence: torch.Tensor):
        assert sequence.ndim == 1
        return "".join(self.vocab.decode(sequence.numpy()))

    def show_samples(self, num=3):
        print(self.__class__.__name__, dict(show_samples=num))
        indices = np.random.choice(len(self), size=num, replace=False)
        for i in indices:
            sequence = self.tensor[i, :]
            print(dict(text=self.sequence_to_text(sequence), raw=sequence))

    def extract_quotes(self, lines: List[str]) -> List[str]:
        def check_speaker_start(s: str) -> bool:
            for char in s:
                if char.isupper():
                    pass
                elif char == ":":
                    return True
                else:
                    return False
            return False

        def handle_newlines(_lines: List[str]) -> List[str]:
            out = []
            for line in _lines:
                out.extend(line.split(self.sep_line))
            return out

        def check_line_finish(s: str) -> bool:
            return s[-1] in "!.?"

        lines = handle_newlines(lines)
        quotes = []
        for row in csv.reader(lines, delimiter=",", quotechar='"'):
            for part in row:
                if not part:
                    continue
                if not check_line_finish(part):
                    continue
                if not check_speaker_start(part):
                    continue
                quotes.append(part)
        if self.hparams.verbose:
            print(dict(extract_quotes=len(quotes)))
        return quotes
