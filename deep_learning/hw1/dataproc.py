from functools import partial
from pathlib import Path

import numpy as np
from sklearn import model_selection


def load_features(folder):
  _fnames = []
  features = []
  for f in Path(folder).iterdir():
    assert f.suffix == ".npy"
    features.append(np.load(f))
    fname = f.stem.split("_")[0]
    assert fname.endswith(".jpg")
    _fnames.append(fname)

  x = np.stack(features)
  print("Features:", x.shape)
  return _fnames, x


def filter_data(x, y, idx2label, labels_keep):
  label2idx = {v: k for k, v in idx2label.items()}
  col_idxs_keep = [label2idx[label] for label in labels_keep]
  cols_keep = y[:, col_idxs_keep]

  mask_keep = np.sum(cols_keep, axis=1) > 0
  x = x[mask_keep]
  y = y[mask_keep][:, col_idxs_keep]
  print("Filtered data:", x.shape, y.shape)
  return x, y


def load_labels(_fnames, txt_concepts, txt_annotations):
  idx2label = {}
  with open(txt_concepts) as f:
    header = f.readline()
    for line in f:
      number, name = line.strip().split()
      number = int(number)
      idx2label[number] = name
  print("Num labels:", len(idx2label))

  y = [-1] * len(_fnames)
  fname2idx = {fname: idx for idx, fname in enumerate(_fnames)}
  with open(txt_annotations) as f:
    for line in f:
      line = line.strip()
      fname, label_string = line.split(maxsplit=1)
      label = [int(_) for _ in label_string.split()]
      idx = fname2idx[fname]
      y[idx] = label
  assert all([_ is not None for _ in y])
  y = np.asarray(y)
  return idx2label, y


def check_onehot(y):
  # Mutually exclusive
  n_samples, n_classes = y.shape
  assert np.array_equal(np.sum(y, axis=-1), np.ones(n_samples))


def split_data(x, y, splits):
  splits = np.asarray(splits) / np.sum(splits)
  assert len(splits) > 1
  assert len(x) == len(y)
  split_fn = partial(model_selection.train_test_split,
                     stratify=y,
                     random_state=42)

  if len(splits) == 2:
    x1, x2, y1, y2 = split_fn(x, y, test_size=splits[1])
    return [x1, x2], [y1, y2]

  else:
    x1, x2, y1, y2 = split_fn(x, y, test_size=sum(splits[1:]))
    xs, ys = split_data(x2, y2, splits[1:])
    return [x1] + xs, [y1] + ys


def save_arrays(arrays, names):
  assert len(arrays) == len(names)
  for a, n in zip(arrays, names):
    np.save(n + ".npy", a)


if __name__ == "__main__":
  fnames, x = load_features("imagecleffeats/imageclef2011_feats")
  idx2label, y = load_labels(fnames, "concepts_2011.txt",
                             "trainset_gt_annotations.txt")
  seasons = ["Spring", "Summer", "Autumn", "Winter"]
  x, y = filter_data(x, y, idx2label, seasons)
  check_onehot(y)
  y = np.argmax(y, axis=-1)  # one-hot to categorical
  print("Final x/y:", x.shape, y.shape)

  xs, ys = split_data(x, y, splits=[0.6, 0.25, 0.15])
  names = "x_trn x_val x_test y_trn y_val y_test".split()
  save_arrays([*xs, *ys], names)

  print("\nClass counts for trn/val/test")
  for _ in ys:
    print(np.unique(_, return_counts=True))
