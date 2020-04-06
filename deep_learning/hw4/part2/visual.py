from main import PickleSaver
import pandas as pd
import matplotlib.pyplot as plt


def invert_two_level_dict(d):
    new = {}
    for key_outer in d.keys():
        for key_inner, val in d[key_outer].items():
            if key_inner not in new.keys():
                new[key_inner] = {}
            new[key_inner][key_outer] = val
    return new


def main(save_path):
    saver = PickleSaver(save_path)
    name = save_path.split(".")[0]
    results = saver.load()
    metrics = list(map(invert_two_level_dict, results["metrics"]))
    for metric_name in metrics[0].keys():
        print("Metric:", metric_name)
        df = pd.DataFrame([m[metric_name] for m in metrics])
        df.plot(title=f"Model performance")
        plt.ylabel(metric_name)
        plt.xlabel("epoch")
        plt.savefig(name + metric_name + ".png")
        plt.show()
        print(df)


if __name__ == "__main__":
    main("results1.pkl")
    main("results2.pkl")
    main("results3.pkl")
