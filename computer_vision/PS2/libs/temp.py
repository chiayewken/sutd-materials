import numpy as np


def get_size_out(size_in, k, s, p):
    assert (size_in - k + 2 * p) % s == 0
    return (size_in - k + 2 * p) // s + 1


def convolve(image, filter, stride, pad):
    image = np.pad(image, pad_width=((0, 0), (pad, pad), (pad, pad)), mode="constant")
    c, h, w = image.shape
    f, c2, h_k, w_k = filter.shape
    assert c == c2
    filter = np.reshape(filter, (f, c, h_k * w_k))
    filter = np.transpose(filter, (2, 1, 0))
    h_out = get_size_out(h, h_k, stride, p=0)
    w_out = get_size_out(w, w_k, stride, p=0)
    out = np.zeros((f, h_out, w_out), dtype=image.dtype)

    for i_out, i_in in enumerate(range(0, h - h_k, stride)):
        for j_out, j_in in enumerate(range(0, w - w_k, stride)):
            window = image[:, i_in : i_in + h_k, j_in : j_in + w_k]
            window = np.reshape(window, (1, c, h_k * w_k))
            window = np.transpose(window, (2, 0, 1))
            window = np.matmul(window, filter)
            window = np.squeeze(window)
            window = np.sum(window, axis=0)
            out[:, i_out, j_out] = window
    return out


image = np.zeros((3, 32, 32))
filter = np.zeros((16, 3, 5, 5))
print(convolve(image, filter, stride=1, pad=2).shape)
