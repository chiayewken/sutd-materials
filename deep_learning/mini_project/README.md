### Usage
To repeat the results (Tested on Ubuntu, Python 3.6.9):
```
pip install -r requirements.txt
python pascal_classify.py
```
### Data
The dataset is from the VOCPascal 2012 Object Classification Competition.
The task is multi-label image classification. During image pre-processing,
the images were center-cropped to 224x224, and normalized with mean values of
[0.485, 0.456, 0.406] and standard deviation values of [0.229, 0.224, 0.225].
Image augmentation was used, and the images were randomly flipped horizontally.

### Model
The model architecture chosen for the classifier is a ResNet architecture as
it is a strong benchmark on many computer vision tasks. The final layer is a 
linear layer with 20 output units and sigmoid activation.

### Loss
For the multi-label classification task, the loss function chosen is binary cross-entropy. This loss
is chosen for multi-label classification because unlike softmax, each label component is
independent from the others. Specifically, this refers to the PyTorch BCELoss. For evaluation
metrics, the mean average precision and tail accuracy were recorded individually for each class
label and also aggregated by the mean across all labels.

### Training procedures
All layers of the model was used for training, for a maximum
of 100 epochs with early stopping by monitoring the validation loss per
epoch. The model with the best validation loss is saved and used for evaluation.

### Hyper-parameters
The optimizer used is Adam with learning rate 1e-4. The batch size for training
and validation data loaders is 32. For reproducibility, the random seed for PyTorch
and NumPy was fixed to a value of 0.

### Experiments
|Setting|Loss|Precision|
|---|---|---|
|ResNet-18 (random initialization)|0.183974|0.350920|
|ResNet-18 (pre-trained initialization)|0.093808|0.786182|
|**ResNet-50 (pre-trained initialization)**|**0.086262**|**0.805692**|

### Evaluation results
The model used is from the experiment setting with best results: 
ResNet-50 (pre-trained initialization)

Label-wise loss and average precision measure:

|          label |     loss|  precision|
|---|---|---|
|     aeroplane | 0.027057 |  0.976711|
|       bicycle | 0.071877  | 0.811710|
|          bird | 0.050317   |0.930293|
|          boat | 0.056896  | 0.839253|
|        bottle | 0.125592  | 0.612504 |
|           bus | 0.035064  | 0.912624 |
|           car | 0.159845  | 0.759892|
|           cat | 0.060098  | 0.954690 |
|         chair | 0.173884  | 0.676514 |
|           cow | 0.051159  | 0.710699|
|  diningtable | 0.093069  | 0.619148 |
|          dog | 0.103512  | 0.906375|
|        horse | 0.050691  | 0.877835|
|    motorbike | 0.059393  | 0.862440 |
|       person | 0.247739  | 0.940645  |
|  pottedplant | 0.119417  | 0.515854  |
|        sheep | 0.034236  | 0.875142 |
|         sofa | 0.093694  | 0.602204 |
|        train | 0.035578  | 0.941048 |
|    tvmonitor | 0.076128  | 0.788262|

Top 5 images for ['train', 'bicycle', 'tvmonitor', 'chair', 'diningtable']:
![](https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/plot_rank_5_reverse_False.png)

Bottom 5 images for ['train', 'bicycle', 'tvmonitor', 'chair', 'diningtable']:
![](https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/plot_rank_5_reverse_True.png)

Tail accuracy plot (average over all labels):
![](https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/tail_acc_plot.png)