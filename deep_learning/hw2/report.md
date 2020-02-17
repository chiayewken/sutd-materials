### Why classwise/stratified split? 
Want train/val/test class distribution to be consistent so acc 
score on train can be meaningfully extrapolated to val/test data.
Don't want to bias model to distribution that is not representative 
of original (unsplit) data. Preserve proportional representation of 
minority/majority classes. Decrease variability in class distribution between splits.

### Tuning regularization
* C=0.001, val_acc=0.3625123098807309
* C=0.01, val_acc=0.519581828792355
* C=0.1, val_acc=0.4931885326622169
* C=0.31622776601683794, val_acc=0.4923655943392786
* C=1, val_acc=0.4923655943392786
* C=3.1622776601683795, val_acc=0.4923655943392786
* C=10, val_acc=0.4923655943392786
* C=31.622776601683793, val_acc=0.4923655943392786

### Model performance using best regularization constant: 0.01

#### Fit model on train data only
* Final val scores
    * Vanilla acc: 0.7248520710059172
    * Class-avg acc: 0.519581828792355
* Final test scores
    * Vanilla acc: 0.7205882352941176
    * Class-avg acc: 0.5337142303908229

#### Fit model on train + val data
* Final val scores
    * Vanilla acc: 0.893491124260355
    * Class-avg acc: 0.7524665353612722
* Final test scores
    * Vanilla acc: 0.7598039215686274
    * Class-avg acc: 0.5975598427916519