import matplotlib.pyplot as plt
import pandas as pd

from task2_train import PickleSaver


def invert_two_level_dict(d):
    new = {}
    for key_outer in d.keys():
        for key_inner, val in d[key_outer].items():
            if key_inner not in new.keys():
                new[key_inner] = {}
            new[key_inner][key_outer] = val
    return new


def main():
    save_path = "results.pkl"
    saver = PickleSaver(save_path)
    results = saver.load()
    metrics = list(map(invert_two_level_dict, results["metrics"]))
    for name in metrics[0].keys():
        print("Metric:", name)
        df = pd.DataFrame([m[name] for m in metrics])
        df.plot(title=f"Model performance for learning rate={results['learn_rate']}")
        plt.ylabel(name)
        plt.xlabel("epoch")
        plt.savefig(name + ".png")
        plt.show()
        print(df)


if __name__ == "__main__":
    main()
