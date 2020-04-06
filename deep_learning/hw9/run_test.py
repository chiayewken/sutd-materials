from run_train import CharGenerationSystem, TrainResult


def main(path_results="train_results.pt"):
    results = TrainResult.batch_load(path_results)
    # for r in results:
    #     system = CharGenerationSystem(r.hparams)
    #     system.net.load_state_dict(r.weights)
    #     system.sample(length=100)
    #     print(r.history[-1])

    print(TrainResult.batch_summary(results))


if __name__ == "__main__":
    main()
