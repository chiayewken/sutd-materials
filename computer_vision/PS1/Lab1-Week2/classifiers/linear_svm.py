import numpy as np
from random import shuffle


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).
    
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TOD:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    pass

    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        probs = np.matmul(X[i], W)

        for j in range(W.shape[-1]):
            if j == y[i]:
                continue
            margin = probs[j] - probs[y[i]] + 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] = dW[:, y[i]] - X[i]
                dW[:, j] = dW[:, j] + X[i]

    loss = (loss / num_train) + (reg * np.sum(np.square(W)))
    dW = (dW / num_train) + (reg * 2 * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    
    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TOD:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    pass
    num_train = X.shape[0]
    probs = np.matmul(X, W)
    margin = np.maximum(0, probs - probs[np.arange(num_train), y][:, np.newaxis] + 1)
    margin[np.arange(num_train), y] = 0

    loss = (margin.sum() / num_train) + (reg * np.sum(np.square(W)))
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TOD:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass

    margin[margin > 0] = 1
    margin[np.arange(num_train), y] -= margin.sum(axis=1)
    dW = ((X.T).dot(margin) / num_train) + (reg * 2 * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
