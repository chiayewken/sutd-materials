from reptile import get_hparams_intent, MetaLearnSystem


def main():
    hp = get_hparams_intent(algo="prototype")
    system = MetaLearnSystem(hp)
    system.run_train()


if __name__ == "__main__":
    main()
