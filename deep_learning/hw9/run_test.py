from datasets import StarTrekCharGenerationDataset
from run_train import CharGenerationSystem, ResultsManager


def main(path_results="results_train.pt"):
    manager = ResultsManager(path_results)
    best = manager.get_best()
    print(dict(num_epochs=len(best.history)))
    print(best.hparams)
    system = CharGenerationSystem.load(best.hparams, best.weights)
    dataset: StarTrekCharGenerationDataset = system.datasets["train"]
    for quote in dataset.extract_quotes(system.sample(length=200)):
        print(quote)
    print(manager.get_summary())


if __name__ == "__main__":
    main()
