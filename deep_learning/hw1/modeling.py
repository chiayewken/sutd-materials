import numpy as np
from sklearn import svm


class OVRLinearSVC:
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.models = None

  def fit(self, x, y):
    n_labels = len(np.unique(y))
    assert n_labels == np.max(y) + 1
    # self.models = [svm.LinearSVC(**self.kwargs) for i in range(n_labels)]
    self.models = [
        svm.SVC(kernel="linear", **self.kwargs) for i in range(n_labels)
    ]
    for i in range(n_labels):
      _y = np.zeros_like(y)
      _y[y == i] = 1
      self.models[i].fit(x, _y)

  def predict(self, x):
    probs = np.zeros(shape=(len(x), len(self.models)))
    for i, _model in enumerate(self.models):
      probs[:, i] = _model.decision_function(x)
    preds = np.argmax(probs, axis=-1)
    return preds


def acc_vanilla(y_true, y_pred):
  assert len(y_true) == len(y_pred)
  return np.sum(np.equal(y_true, y_pred)) / len(y_true)


def acc_class_averaged(y_true, y_pred):
  scores = {}
  for label in np.unique(y_true):
    mask = (y_true == label)
    scores[label] = acc_vanilla(y_true[mask], y_pred[mask])
  print("Class wise acc:", scores)
  return np.mean(list(scores.values()))


def tune_reg(data_trn, data_val, values_reg):
  print("\nTuning regularization")
  _x_val, _y_val = data_val
  scores = []
  for val in values_reg:
    _model = OVRLinearSVC(C=val)
    _model.fit(*data_trn)
    scores.append(acc_class_averaged(_y_val, _model.predict(_x_val)))
    print(f"C={val}, val_acc={scores[-1]}")
  reg_best = values_reg[np.argmax(scores)]
  print("Best regularization value:", reg_best)
  return reg_best


def concat(*arrays):
  return np.concatenate(arrays, axis=0)


def load_arrays(names):
  return [np.load(n + ".npy") for n in names]


if __name__ == "__main__":
  names = "x_trn x_val x_test y_trn y_val y_test".split()
  x_trn, x_val, x_test, y_trn, y_val, y_test = load_arrays(names)

  # model = MulticlassLinearSVC()
  # model.fit(x_trn, y_trn)
  # print(acc_vanilla(y_val, model.predict(x_val)))
  # print(acc_class_averaged(y_val, model.predict(x_val)))

  vals_reg = [0.001, 0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 1000**0.5]
  val_reg_best = tune_reg((x_trn, y_trn), (x_val, y_val), vals_reg)

  print("Model performance using best regularization constant:", val_reg_best)
  print()
  model = OVRLinearSVC(C=val_reg_best)
  print("Fit model on train data only")
  model.fit(x_trn, y_trn)
  print("Final val scores")
  print("Vanilla acc:", acc_vanilla(y_val, model.predict(x_val)))
  print("Class-avg acc:", acc_class_averaged(y_val, model.predict(x_val)))
  print("Final test scores")
  print("Vanilla acc:", acc_vanilla(y_test, model.predict(x_test)))
  print("Class-avg acc:", acc_class_averaged(y_test, model.predict(x_test)))
  print()
  print("Fit model on train + val data")
  model.fit(concat(x_trn, x_val), concat(y_trn, y_val))
  print("Final val scores")
  print("Vanilla acc:", acc_vanilla(y_val, model.predict(x_val)))
  print("Class-avg acc:", acc_class_averaged(y_val, model.predict(x_val)))
  print("Final test scores")
  print("Vanilla acc:", acc_vanilla(y_test, model.predict(x_test)))
  print("Class-avg acc:", acc_class_averaged(y_test, model.predict(x_test)))
