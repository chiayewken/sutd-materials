# Generating Star Trek Quotes: Character-level language modeling
Chia Yew Ken (1002675)
## Usage
* To train from scratch instead of loading saved results, do ```rm -rf *.pt``` first
* This system was tested in Google Colab environment (Ubuntu 18.04, Python 3.6.9, Tesla K80 GPU)
```
pip install -r requirements.txt
python run_train.py
python run_results.py
```
## System
* ```run_train.py``` runs a grid search over hyper-parameters.
* The grid search was reduced to 1 epoch per experiment in the interest of time.
* The full results are in the appendix below.
* The best hyper-parameters are used to train a new model to convergence with early-stopping 
based on validation loss.
* The architectures tested include Long-Short-Term-Memory, 
Gated-Recurrent-Unit and Temporal-Convolutional-Networks
* The best model was a LSTM with 3 layers and 256 hidden units, trained with batch size of 512.
```
grid = dict(
    model=["lstm", "gru", "tcn"],
    n_layers=[1, 2, 3],
    n_hidden=[128, 256],
    bs=[32, 128, 512],
)
``` 
## Results
* The best quote found was: ```"SPOCK: The procedure is correct. The love is satisfactory."```
* The saved checkpoints for hyper-parameter search: ```results_search.pt``` and final model: ```results_train.pt```
* The .txt file with generated Star Trek quotes: ```outputs/samples.txt```
```
{
    "0": [
        "KIRK: What happened to the close which of the contring what he has a been commanded out of the planet as supply and a centre of the confident of the captain.",
        "KIRK: Well, I would have dead and better them out of a honour transporter room. I don't kno and all right to be able to beam up to be dead.",
        "SCOTT: Well, you can't be a man warp to the mission, t and the Enterprise, but you are in a hundred of the control and the fight of the ship here with a weapon as a form of the centre of the two other hands of the serpent four stations. They can see you  perfection and been a chance and starts real the ship to the captain of the computer. They said it off and leaves the creature, Mister Spock.",
        "SPOCK: The engineering and deal when the other secret oe as a three moment of here.",
        "KIRK: Yes, sir."
    ],
    "1": [
        "SPOCK: The captain will rig to the planet and the communicator banks of storm. The other time to be travelled by fact, what primitive is true to the transporter room. The development.",
        "KIRK: Bones, how do you think we're to expect as we could have a command of this planet?",
        "KIRK: Don't you turn about the first time, then he has a friendly shields, and you are a ld the time.,KIRK: The memory of this ship have been destroyed him. I don't know. I do not know.",
        "KIRK: And a man always to him.",
        "KIRK: Mister Spock, my orders are. I want to know anything you don't know.,KIRK: It's considered. W should be in the transporter beam down and all this planet."
    ],
    "2": [
        "MCCOY: The most strangers with him, Jim. Spock with two distress call.",
        "KIRK: You'll kill him.",
        "KIRK: Specify.",
        "SULU: Aye, sir.nd enough to you.",
        "SPOCK: Doctor, how do you think you can know what was happening to you."
    ],
    "3": [
        "KIRK: Miste and I want to deal with the ship of this close and their computer, and a lot of the bridge. He's the purpose of the secret. A point of the table and he throws him a leader starts to be out of his sto the planet at the contrary. The man provided to security and four hours, there's a false to human to handle it. The more inferior without symptons of the blower start operations.",
        "KIRK: Lieutenant, es the last time to be seen the door.",
        "KIRK: I'm sorry, Spock, but we can control you and the computer completely absorbed the planet is not a paralysis of this moment. The transfer is much as we shoat a truly better choice. They won't be able to prove it now.",
        "MCCOY: Yes. I shall be in the ground the human beings. The computer room is now in the mind-try of the portal to the one solid reading ue to be a short from the prisoner of energy beam down.",
        "KIRK: Good. The power for the starship commander so much as I thought you know that you will follow your ability to be a few more of the group ing a sample flight but produces the ship is a remote to the captain."
    ],
    "4": [
        "SULU: For seven point three minutes.",
        "KIRK: He was beamed down the shuttlecraft.",
        "KIRK: You don't know, sir.",
        "KIRK: What do you mean, it's busy. You can feel all right, please and hear the creature is a corridor. We can communicate with the phaser open did we have to do is the transporter room.",
        "KIRK: Yes, it would be a few hours, and you didn't have anything else to stay away?"
    ],
    "5": [
        "KIRK: Come on. What is it?",
        "KIRK: All right, Mister Spock, we hav ship is still alive.",
        "KIRK: You mean to the ship's destruction of the Enterprise, and I want you to continue a course for ower the comet waves on force and the starboard ship will be able to get to the ship. Thank you. They're destroying you now, sir.",
        "SPOCK: She's a security officer, Mister Spock. So I'll take the ship to shields.,KIRK: We are not serious the antimatter reading. I could have seen anything we had to control the Enterprise. Interesting. Beam down to Sickbay. My hobbys are still approaching the clouds the devil is in contact with the computer concluded of the star system and the others are standing on the ship. The answers only one of the androids who have been sent to understand. I am pleased theain.",
        "KIRK: Yes, why don't you still have the captain. It is dead."
    ],
    "6": [
        "KIRK: Get this ship.",
        "SPOCK: It is not too long. It is our console clock shoots an area of our planet, and they're still there.",
        "SCOTT: Aye, sir.",
        "MCCOY: Well, there was a reason for you and the timelon engine gods?",
        "KIRK: Thank you, Mister Spock. It should have been expressed with the transporter room."
    ],
    "7": [
        "KIRK: But not three years, the same spot. You will be coming anyway. (sings even final container a weak an immediate rest of the ship and the same time by this galaxy.",
        "KIRK: You have located the ship is under attack by the council minutes, there are those qualities or some areas of the moment.",
        "MCCOY: Why, millions of this ship. And the main controls. It goes out of bottles. It is being like an attention, the proper time. There's something wroll the man who have seen this?",
        "KIRK: It's all right.",
        "SPOCK: I am not a starship. We're here to beam up."
    ]
}
```
* The plot for loss: ```outputs/loss.jpg```, accuracy: ```outputs/acc.jpg```
* The table of results per epoch for the final model: ```outputs/metrics.txt```

|    |   test_acc |   test_loss |   train_acc |   train_loss |   val_acc |   val_loss |
|----|------------|-------------|-------------|--------------|-----------|------------|
|  0 |      0.583 |       1.366 |       0.536 |        1.551 |     0.581 |      1.367 |
|  1 |      0.599 |       1.312 |       0.603 |        1.282 |     0.598 |      1.312 |
|  2 |      0.603 |       1.3   |       0.619 |        1.221 |     0.603 |      1.297 |
|  3 |      0.607 |       1.293 |       0.63  |        1.18  |     0.604 |      1.295 |
|  4 |      0.607 |       1.301 |       0.64  |        1.146 |     0.605 |      1.301 |
|  5 |      0.605 |       1.321 |       0.648 |        1.118 |     0.606 |      1.317 |
|  6 |      0.605 |       1.344 |       0.656 |        1.091 |     0.602 |      1.343 |
|  7 |      0.603 |       1.361 |       0.663 |        1.067 |     0.602 |      1.356 |

## Appendix
* Full grid search results: ```outputs/search_table.txt```

|   bs | model   |   n_hidden |   n_layers |   test_acc |   test_loss |   train_acc |   train_loss |   val_acc |   val_loss |
|------|---------|------------|------------|------------|-------------|-------------|--------------|-----------|------------|
|   32 | tcn     |        256 |          1 |      0.393 |       3.193 |       0.313 |       14.981 |     0.377 |      2.902 |
|  128 | tcn     |        256 |          1 |      0.404 |       2.878 |       0.32  |       16.115 |     0.402 |      2.725 |
|   32 | tcn     |        128 |          1 |      0.39  |       2.686 |       0.292 |       10.823 |     0.383 |      2.522 |
|  512 | tcn     |        256 |          1 |      0.437 |       2.403 |       0.353 |       13.784 |     0.434 |      2.388 |
|  128 | tcn     |        128 |          1 |      0.416 |       2.371 |       0.319 |       10.711 |     0.423 |      2.352 |
|  512 | tcn     |        128 |          1 |      0.431 |       2.31  |       0.323 |        9.734 |     0.426 |      2.323 |
|   32 | tcn     |        256 |          3 |      0.404 |       2.258 |       0.339 |        5.384 |     0.395 |      2.188 |
|   32 | tcn     |        128 |          3 |      0.412 |       2.178 |       0.342 |        4.219 |     0.413 |      2.183 |
|   32 | tcn     |        128 |          2 |      0.406 |       2.224 |       0.334 |        4.786 |     0.399 |      2.173 |
|  128 | tcn     |        256 |          2 |      0.412 |       2.165 |       0.372 |        5.429 |     0.409 |      2.167 |
|   32 | tcn     |        256 |          2 |      0.423 |       2.137 |       0.35  |        5.69  |     0.405 |      2.123 |
|  512 | tcn     |        128 |          3 |      0.4   |       2.091 |       0.379 |        3.781 |     0.397 |      2.091 |
|  128 | tcn     |        128 |          2 |      0.464 |       1.942 |       0.382 |        4.305 |     0.458 |      1.932 |
|  512 | tcn     |        128 |          2 |      0.465 |       1.922 |       0.386 |        4.179 |     0.459 |      1.921 |
|  512 | tcn     |        256 |          2 |      0.464 |       1.955 |       0.388 |        5.279 |     0.465 |      1.91  |
|  128 | tcn     |        128 |          3 |      0.461 |       1.904 |       0.375 |        3.94  |     0.464 |      1.885 |
|  512 | tcn     |        256 |          3 |      0.455 |       1.892 |       0.392 |        4.814 |     0.456 |      1.884 |
|  128 | tcn     |        256 |          3 |      0.473 |       1.911 |       0.377 |        4.995 |     0.47  |      1.88  |
|   32 | gru     |        256 |          1 |      0.465 |       1.813 |       0.429 |        2.047 |     0.457 |      1.847 |
|   32 | gru     |        128 |          1 |      0.45  |       1.854 |       0.42  |        2.07  |     0.45  |      1.846 |
|   32 | lstm    |        128 |          1 |      0.469 |       1.768 |       0.438 |        1.96  |     0.472 |      1.77  |
|   32 | gru     |        128 |          2 |      0.475 |       1.767 |       0.439 |        1.948 |     0.486 |      1.754 |
|   32 | gru     |        128 |          3 |      0.486 |       1.695 |       0.443 |        1.924 |     0.48  |      1.719 |
|  128 | gru     |        128 |          1 |      0.491 |       1.735 |       0.446 |        1.962 |     0.495 |      1.7   |
|   32 | lstm    |        128 |          2 |      0.49  |       1.691 |       0.448 |        1.911 |     0.491 |      1.691 |
|   32 | lstm    |        128 |          3 |      0.494 |       1.65  |       0.441 |        1.936 |     0.495 |      1.689 |
|   32 | lstm    |        256 |          1 |      0.493 |       1.682 |       0.459 |        1.855 |     0.493 |      1.686 |
|   32 | gru     |        256 |          2 |      0.497 |       1.668 |       0.453 |        1.894 |     0.489 |      1.683 |
|   32 | gru     |        256 |          3 |      0.511 |       1.623 |       0.46  |        1.86  |     0.506 |      1.653 |
|  128 | lstm    |        128 |          1 |      0.504 |       1.68  |       0.463 |        1.857 |     0.505 |      1.648 |
|  512 | gru     |        128 |          1 |      0.517 |       1.631 |       0.467 |        1.872 |     0.514 |      1.625 |
|  128 | gru     |        256 |          1 |      0.504 |       1.654 |       0.471 |        1.845 |     0.511 |      1.622 |
|  128 | gru     |        128 |          2 |      0.515 |       1.616 |       0.469 |        1.821 |     0.519 |      1.604 |
|   32 | lstm    |        256 |          2 |      0.522 |       1.558 |       0.48  |        1.763 |     0.514 |      1.596 |
|  512 | lstm    |        128 |          1 |      0.531 |       1.583 |       0.483 |        1.781 |     0.528 |      1.577 |
|  128 | gru     |        128 |          3 |      0.529 |       1.589 |       0.475 |        1.794 |     0.527 |      1.571 |
|  128 | lstm    |        128 |          2 |      0.526 |       1.589 |       0.476 |        1.798 |     0.527 |      1.569 |
|   32 | lstm    |        256 |          3 |      0.524 |       1.538 |       0.479 |        1.771 |     0.511 |      1.569 |
|  128 | lstm    |        128 |          3 |      0.529 |       1.583 |       0.472 |        1.82  |     0.53  |      1.553 |
|  128 | lstm    |        256 |          1 |      0.531 |       1.563 |       0.492 |        1.724 |     0.535 |      1.539 |
|  512 | gru     |        256 |          1 |      0.539 |       1.538 |       0.495 |        1.743 |     0.536 |      1.536 |
|  512 | gru     |        128 |          2 |      0.543 |       1.529 |       0.492 |        1.731 |     0.54  |      1.527 |
|  128 | gru     |        256 |          2 |      0.532 |       1.551 |       0.493 |        1.725 |     0.54  |      1.524 |
|  512 | lstm    |        128 |          2 |      0.546 |       1.509 |       0.498 |        1.712 |     0.544 |      1.504 |
|  512 | lstm    |        128 |          3 |      0.549 |       1.504 |       0.491 |        1.745 |     0.549 |      1.503 |
|  512 | gru     |        128 |          3 |      0.548 |       1.499 |       0.498 |        1.706 |     0.544 |      1.499 |
|  128 | gru     |        256 |          3 |      0.545 |       1.505 |       0.499 |        1.699 |     0.544 |      1.483 |
|  512 | lstm    |        256 |          1 |      0.556 |       1.466 |       0.518 |        1.626 |     0.556 |      1.465 |
|  512 | gru     |        256 |          2 |      0.559 |       1.452 |       0.516 |        1.635 |     0.557 |      1.451 |
|  128 | lstm    |        256 |          2 |      0.558 |       1.466 |       0.512 |        1.635 |     0.56  |      1.447 |
|  512 | gru     |        256 |          3 |      0.569 |       1.427 |       0.519 |        1.622 |     0.566 |      1.426 |
|  128 | lstm    |        256 |          3 |      0.561 |       1.446 |       0.514 |        1.633 |     0.564 |      1.421 |
|  512 | lstm    |        256 |          2 |      0.578 |       1.382 |       0.538 |        1.54  |     0.578 |      1.383 |
|  512 | lstm    |        256 |          3 |      0.584 |       1.365 |       0.536 |        1.551 |     0.582 |      1.363 |

Final model full hyper-parameters
```
{'root': 'data', 'lr': 0.001, 'bs': 512, 'steps_per_epoch': 1000, 'epochs': 100
, 'n_hidden': 256, 'model': 'lstm', 'n_layers': 3, 'dropout': 0.0, 'tie_embed_weights': True, 'seq_len':
 32, 'dev_run': False, 'verbose': True}
```