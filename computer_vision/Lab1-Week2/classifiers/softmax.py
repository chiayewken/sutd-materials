import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    
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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    def stable_softmax(x):
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def crossentropy_loss(y_true, y_pred):
        return -np.log(y_pred[y_true])

    outputs = np.matmul(X, W)  # N, C
    for i in range(len(outputs)):
        probs = stable_softmax(outputs[i])

        # Crossentropy loss
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass

    def stable_softmax(logits):
        exps = np.subtract(logits, np.max(logits, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)

    logits = np.matmul(X, W)  # N, C
    softmax = stable_softmax(logits)
    loss = np.mean(-np.log(softmax[:, y]))

    softmax[:, y] -= 1
    grad = np.matmul(np.transpose(X), softmax)
    grad /= len(X)

    loss += reg * np.sum(np.square(W))
    grad += reg * 2 * W
    dW = grad

    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
