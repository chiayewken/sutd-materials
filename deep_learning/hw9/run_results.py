import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from datasets import Splits
from run_train import ResultsManager, TrainResult, CharGenerationSystem


def do_tabulate(df: pd.DataFrame, show_index=True):
    return tabulate(df, tablefmt="github", headers="keys", showindex=show_index)


def plot_metrics(result: TrainResult, output_dir: Path):
    for key in ["loss", "acc"]:
        for s in [Splits.train, Splits.val, Splits.test]:
            plt.plot([h[s][key] for h in result.history], label=s + "_" + key)
        plt.ylabel(key)
        plt.xlabel("epochs")
        plt.legend()
        plt.savefig(str((output_dir / key).with_suffix(".jpg")))
        plt.clf()

    df_metrics = result.get_metrics()
    with open(str(output_dir / "metrics.txt"), "w") as f:
        f.write(do_tabulate(df_metrics))


def write_quotes(result: TrainResult, output_dir: Path):
    data = {i: h["quotes"] for i, h in enumerate(result.history)}
    with open(str(output_dir / "samples.txt"), "w") as f:
        f.write(json.dumps(data, indent=4))


def write_search_table(path_results: str, output_dir: Path):
    manager = ResultsManager(path_results)
    summary = manager.get_summary()
    with open(str(output_dir / "search_table.txt"), "w") as f:
        f.write(do_tabulate(summary, show_index=False))


def main(
    path_results_search="results_search.pt",
    path_results_train="results_train.pt",
    output_dir="outputs",
):
    manager = ResultsManager(path_results_train)
    result = manager.get_best()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    write_quotes(result, output_dir)
    plot_metrics(result, output_dir)
    write_search_table(path_results_search, output_dir)

    system = CharGenerationSystem.load(result.hparams, result.weights)
    print(system.device)
    print(system.sample_quotes())


if __name__ == "__main__":
    main()
