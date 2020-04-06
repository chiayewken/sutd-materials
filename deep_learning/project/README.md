# Learning to learn robust intent classification

## Introduction
Dialog systems such as chat-bots allow humans to naturally interact with digital systems.
Concretely, intent classification aims to categorize dialog text into a fixed 
set of labels in order for the dialog system to produce the appropriate response.
This is closely related to general text classification in Natural Language Processing.

However, there are some specific challenges:
* There may be many intent labels and some may be closely related
* Out-of-scope queries are inevitable, and detection is necessary
 to avoid derailing the user experience
* Training data is usually limited eg 100 samples per label
* The classifier should be able to accept new labels in the future

Compared to machine learning models, humans are able to learn quickly, 
and generalize well even with little training data.
Humans are also robust and able to detect anomalous data more easily.
However, this ability is not completely innate to most humans, and must be learnt.
Hence, the goal of a meta-learner is learning to learn any new task quickly.

We propose to apply meta-learning techniques to produce a meta-learner for
for robust intent classification.