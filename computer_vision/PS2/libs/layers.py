from builtins import range
import numpy as np

####x
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    dim_size = x[0].shape
    X = np.reshape(x, (x.shape[0], np.prod(dim_size)))
    out = X.dot(w) + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


####x
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    dim_shape = np.prod(x[0].shape)
    N = x.shape[0]
    X = np.reshape(x, (N, dim_shape))
    # input gradient
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    # weight gradient
    dw = X.T.dot(dout)
    # bias gradient
    db = dout.sum(axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


####x
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    out = np.maximum(x, 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


####x
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    dx = dout * (x > 0).astype(np.float32)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


###x
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        probs = np.random.uniform(0, 1, size=x.shape)
        mask = (probs > p).astype(np.float32)
        out = np.multiply(x, mask)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


###x
def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        dx = np.multiply(dout, mask)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


###x
def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    def get_size_out(size_in, k, s, p):
        assert (size_in - k + 2 * p) % s == 0
        return (size_in - k + 2 * p) // s + 1

    #
    # def convolve(image, filter, stride, pad):
    #     image = np.pad(
    #         image, pad_width=((0, 0), (pad, pad), (pad, pad)), mode="constant"
    #     )
    #     c, h, w = image.shape
    #     f, c2, h_k, w_k = filter.shape
    #     assert c == c2
    #     filter = np.reshape(filter, (f, c, h_k * w_k))
    #     filter = np.transpose(filter, (2, 1, 0))
    #     h_out = get_size_out(h, h_k, stride, p=0)
    #     w_out = get_size_out(w, w_k, stride, p=0)
    #     out = np.zeros((f, h_out, w_out), dtype=image.dtype)
    #
    #     for i_out, i_in in enumerate(range(0, h - h_k, stride)):
    #         for j_out, j_in in enumerate(range(0, w - w_k, stride)):
    #             window = image[:, i_in : i_in + h_k, j_in : j_in + w_k]
    #             window = np.reshape(window, (1, c, h_k * w_k))
    #             window = np.transpose(window, (2, 0, 1))
    #             window = np.matmul(window, filter)
    #             window = np.squeeze(window)
    #             window = np.sum(window, axis=0)
    #             out[:, i_out, j_out] = window
    #     return out
    #
    # lst = []
    # for image in x:
    #     lst.append(convolve(image, w, conv_param["stride"], conv_param["pad"]))
    # out = np.stack(lst)
    # out = np.add(out, np.reshape(b, (-1, 1, 1)))

    pad = conv_param["pad"]
    stride = conv_param["stride"]
    N, C, H, W = x.shape
    F, C, FH, FW = w.shape

    outH = get_size_out(H, FH, stride, pad)
    outW = get_size_out(W, FW, stride, pad)

    out = np.zeros((N, F, outH, outW))

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")
    H_pad, W_pad = x_pad.shape[2], x_pad.shape[3]

    w_row = w.reshape(F, C * FH * FW)  # [F x C*FH*FW]

    x_col = np.zeros((C * FH * FW, outH * outW))  # [C*FH*FW x H'*W']
    for index in range(N):
        count = 0
        for i in range(0, H_pad - FH + 1, stride):
            for j in range(0, W_pad - FW + 1, stride):
                x_col[:, count] = x_pad[index, :, i : i + FH, j : j + FW].reshape(
                    C * FH * FW
                )
                count += 1
        out[index] = (w_row.dot(x_col) + b.reshape(F, 1)).reshape(F, outH, outW)
    x = x_pad
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


###x
def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x_pad, w, b, conv_param = cache
    N, F, outH, outW = dout.shape
    N, C, Hpad, Wpad = x_pad.shape
    FH, FW = w.shape[2], w.shape[3]
    stride = conv_param["stride"]
    pad = conv_param["pad"]

    dx = np.zeros((N, C, Hpad - 2 * pad, Wpad - 2 * pad))
    dw, db = np.zeros(w.shape), np.zeros(b.shape)

    w_row = w.reshape(F, C * FH * FW)  # [F x C*FH*FW]

    x_col = np.zeros((C * FH * FW, outH * outW))  # [C*FH*FW x H'*W']
    for index in range(N):
        out_col = dout[index].reshape(F, outH * outW)  # [F x H'*W']
        w_out = w_row.T.dot(out_col)  # [C*FH*FW x H'*W']
        dx_cur = np.zeros((C, Hpad, Wpad))
        count = 0
        for i in range(0, Hpad - FH + 1, stride):
            for j in range(0, Wpad - FW + 1, stride):
                dx_cur[:, i : i + FH, j : j + FW] += w_out[:, count].reshape(C, FH, FW)
                x_col[:, count] = x_pad[index, :, i : i + FH, j : j + FW].reshape(
                    C * FH * FW
                )
                count += 1
        dx[index] = dx_cur[:, pad:-pad, pad:-pad]
        dw += out_col.dot(x_col.T).reshape(F, C, FH, FW)
        db += out_col.sum(axis=1)

    # assert dx.shape == x.shape
    # assert dw.shape == w.shape
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


###x
def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    N, C, H, W = x.shape
    stride = pool_param["stride"]
    PH = pool_param["pool_height"]
    PW = pool_param["pool_width"]
    outH = 1 + (H - PH) // stride
    outW = 1 + (W - PW) // stride

    out = np.zeros((N, C, outH, outW))
    for index in range(N):
        out_col = np.zeros((C, outH * outW))
        count = 0
        for i in range(0, H - PH + 1, stride):
            for j in range(0, W - PW + 1, stride):
                pool_region = x[index, :, i : i + PH, j : j + PW].reshape(C, PH * PW)
                out_col[:, count] = pool_region.max(axis=1)
                count += 1
        out[index] = out_col.reshape((C, outH, outW))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


###x
def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x, pool_param = cache
    N, C, outH, outW = dout.shape
    H, W = x.shape[2], x.shape[3]
    stride = pool_param["stride"]
    PH, PW = pool_param["pool_height"], pool_param["pool_width"]

    dx = np.zeros(x.shape)

    for index in range(N):
        dout_row = dout[index].reshape(C, outH * outW)
        count = 0
        for i in range(0, H - PH + 1, stride):
            for j in range(0, W - PW + 1, stride):
                pool_region = x[index, :, i : i + PH, j : j + PW].reshape(C, PH * PW)
                max_pool_indices = pool_region.argmax(axis=1)
                dout_cur = dout_row[:, count]
                count += 1
                # pass gradient only through indices of max pool
                dmax_pool = np.zeros(pool_region.shape)
                dmax_pool[np.arange(C), max_pool_indices] = dout_cur
                dx[index, :, i : i + PH, j : j + PW] += dmax_pool.reshape((C, PH, PW))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


###x
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
