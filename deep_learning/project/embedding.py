from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import sentence_transformers
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

from utils import get_device
import os


class SentenceBERT:
    embed_size = 768

    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.name = "bert-base-nli-mean-tokens"
        self.model = None

    def load_model(self):
        if self.model is not None:
            return
        model_dir = self.download()
        self.model = sentence_transformers.SentenceTransformer(
            model_name_or_path=str(model_dir), device=get_device(use_gpu=True)
        )

    def download(self):
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/{}.zip"
        url = url.format(self.name)
        model_dir = self.cache_dir / self.name
        if not model_dir.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(
                url, download_root=str(self.cache_dir), extract_root=str(model_dir)
            )
        assert model_dir.exists()
        return model_dir

    def embed_texts(
        self, texts: List[str], cache_name: str, delete_cache=False
    ) -> np.ndarray:
        name = f"embeds_{self.name}_{cache_name}_{len(texts)}"
        path_cache = (self.cache_dir / name).with_suffix(".npy")
        if path_cache.exists() and delete_cache:
            os.remove(str(path_cache))
        if path_cache.exists():
            return np.load(str(path_cache))

        self.load_model()
        embeds = np.stack(self.model.encode(texts, show_progress_bar=True))
        assert embeds.shape == (len(texts), self.embed_size)
        np.save(str(path_cache), embeds)
        return self.embed_texts(texts, cache_name)


class WordEmbedder:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.path_embeds = self.download()
        self.vocab, self.embeds = self.read()
        self.embed_size = self.embeds.shape[-1]
        self.word2i = self.get_word_mapping()

    def get_word_mapping(self) -> Dict[str, int]:
        return {w: i for i, w in enumerate(self.vocab)}

    def download(self):
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/numberbatch-en-19.08.txt.gz"
        path = self.cache_dir / Path(url).stem
        if not path.exists():
            download_and_extract_archive(url, str(self.cache_dir))
        assert path.exists()
        return path

    def read(self) -> Tuple[np.ndarray, np.ndarray]:
        path_vocab = self.cache_dir / "numberbatch_vocab.npy"
        path_embeds = self.cache_dir / "numberbatch_embeds.npy"
        if path_vocab.exists() and path_embeds.exists():
            return (
                np.load(str(path_vocab), allow_pickle=True),
                np.load(str(path_embeds)),
            )
        with open(str(self.path_embeds)) as f:
            num_embeds, embed_size = f.readline().strip().split()
            num_embeds, embed_size = int(num_embeds), int(embed_size)
            embeds = np.empty(shape=(num_embeds, embed_size), dtype=np.object)
            vocab = np.empty(shape=(num_embeds,), dtype=np.object)
            for i, line in tqdm(enumerate(f), total=num_embeds):
                word, *values = line.strip().split()
                vocab[i] = word
                embeds[i] = values

        assert i == num_embeds - 1
        embeds = embeds.astype(np.float32)
        np.save(str(path_vocab), vocab)
        np.save(str(path_embeds), embeds)
        return self.read()

    def preprocess(self, text: str) -> List[str]:
        words = text.lower().strip().split()
        if not words:
            return ["empty"]
        output = []
        for w in words:
            if w.isdigit():
                output.append("number")
            elif w.endswith("am") or w.endswith("pm"):
                output.append("time")
            elif w not in self.word2i.keys():
                output.append("unknown")
            else:
                output.append(w)
        return output

    def encode(self, text: str) -> List[int]:
        words = self.preprocess(text)
        return [self.word2i[w] for w in words]

    def count_stats(self, texts: List[str]):
        words_all = []
        for t in texts:
            words_all.extend(self.preprocess(t))
        total = len(words_all)
        fraction_unknown = words_all.count("unknown") / total
        fraction_empty = words_all.count("empty") / total
        print(dict(fraction_unknown=fraction_unknown, fraction_empty=fraction_empty))

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.array([self.embeds[self.encode(t)] for t in tqdm(texts)])

    def fit_texts(self, texts: List[str]):
        self.count_stats(texts)
        indices = [i for t in texts for i in self.encode(t)]
        indices = sorted(set(indices))
        # old2new = {old:new for new,old in enumerate(indices)}
        self.vocab = self.vocab[indices]
        self.embeds = self.embeds[indices]
        self.word2i = self.get_word_mapping()
        print(dict(embeds=self.embeds.shape))
