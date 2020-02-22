Besides the code, submit also
* a graph showing train/val/test loss averaged for all minibatches of every
epoch (train 5 epochs at least) for one chosen learning rate
* a graph showing train/val/test accuracy after one epoch training for every
epoch (train 5 epochs at least) for one chosen learning rate

The graph plots are saved in "acc.png" and "loss.png"

* When you train a deep neural net, then you
get after every epoch one model (actually after every minibatch). Why
you should not select the best model over all epochs on the test dataset?

The test set is supposed to provide an unbiased estimate of the model 
performance on unseen data. If we select the best model or do hyperparameter tuning based on performance on test 
data, this is a form of data leakage. Hence the model weights we have selected are not
likely to generalize well. The test set scores we have are likely to be over-optimistic.
Instead, we should train on the train set, select model weights and tune hyperparameters 
on the validation set, and lastly use the test set performance as the final unbiased
estimate of generalization capability of the model.