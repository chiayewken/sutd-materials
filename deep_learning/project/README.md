# Meta-Learning for Few-Shot Text Classification
## Usage
Tested on Ubuntu 18.04, Python 3.6.9
* Install requirements: ```pip3 install -r requirements.txt```
* Train with Reptile algorithm: ```python3 reptile.py```
* Train with Prototypical Network algorithm: ```python3 prototype.py```
* Run interactive user interface demo: ```streamlit run demo.py```

## Overview
Few-shot text classifiation is challenging and current research 
focuses on meta-learning methods, using word embedding features and Prototypical Network variants. 
We compare Prototypical Networks with the Reptile Algorithm and leverage deep sentence
embedding features from pre-trained language models. The models are evaluated on a dialog intent
classification dataset and results show that using sentence embeddings from deep pre-trained language
models is superior to word-embedding features, and Prototypical Networks have a slight
advantage over Reptile.

## Results
|            | Reptile (GLoVE + LSTM) | No meta-learning (Sentence-BERT) | Reptile (Sentence-BERT) | Prototypical Networks (Sentence-BERT) |
|-----------:|------------------------|----------------------------------|-------------------------|---------------------------------------|
| Meta-train | N/A                    | N/A                              | 0.947                   | 0.928                                 |
| Meta-valid | N/A                    | N/A                              | 0.848                   | 0.848                                 |
| Meta-test  | 0.361                  | 0.512                            | 0.848                   | 0.904                                 |

Using GLoVE embeddings instead of Sentence-BERT embeddings results in worse performance.
Hence, sentence representations from pre-trained language model is superior.
This could be due to lack of deep contextual information in word embeddings, 
smaller dimension size (300 vs 768) and word-level tokenization which is vulnerable to 
out-of-vocabulary issues at test time. 

Meta-learning was shown to be crucial in order to generalize well to unseen tasks, 
and Prototypical Networks are slightly superior in performance. 
This is likely due to a simpler inductive bias for classification. 
They do not assume a fixed set of class outputs, instead depending on the support set
to construct prototype representation, while the classification is made by pairwise distance comparison.
This is more flexible than optimization-based approaches like Reptile, where the
model must be re-trained if there are new classes.