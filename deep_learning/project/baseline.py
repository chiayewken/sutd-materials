import torch
from sklearn import svm, linear_model

from datasets import Splits, MetaBatch
from reptile import MetaLearnSystem
from utils import MetricsTracker, HyperParams, accuracy_score


class BaselineSystem(MetaLearnSystem):
    def __init__(self, hp: HyperParams):
        super().__init__(hp)

    def loop_eval(self) -> dict:
        tracker = MetricsTracker(prefix=Splits.val)
        for task in MetaBatch(self.batch_val, torch.device("cpu")).get_tasks():
            x_train, y_train = task.train
            x_test, y_test = task.test
            # model = svm.SVC(kernel="rbf")
            model = linear_model.RidgeClassifier()
            model.fit(x_train.numpy(), y_train.numpy())

            logits_numpy = model.predict(x_test.numpy())
            logits = torch.from_numpy(logits_numpy)
            acc = accuracy_score(logits, y_test)
            tracker.store(dict(acc=acc))
        return tracker.get_average()

    def run_train(self):
        print(self.loop_eval())


def main():
    system = BaselineSystem(HyperParams())
    system.run_train()


if __name__ == "__main__":
    main()
