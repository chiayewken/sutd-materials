import json
import matplotlib.pyplot as plt
import tabulate


def plot_test_results(path_save_results, data_split="test", metric_name="loss"):
    assert metric_name in {"loss", "acc"}
    with open(path_save_results) as f:
        for line in f:
            history = []
            experiment = json.loads(line)
            hparams = experiment["hparams"]
            if hparams["rnn_type"] == "lstm" and hparams["rnn_num_layers"] == 1:
                for result in experiment["history"]:
                    history.append(result[data_split][metric_name])
                plt.plot(history, label=str(dict(batch_size=hparams["batch_size"])))
        plt.ylabel(metric_name)
        plt.xlabel("epochs")
        plt.legend()
        plt.savefig(metric_name + ".jpg")
        plt.show()


def main(path_save_results="results.json"):
    plot_test_results(path_save_results, metric_name="loss")
    plot_test_results(path_save_results, metric_name="acc")


if __name__ == "__main__":
    main()
