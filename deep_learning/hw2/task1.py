import time

import numpy as np
import torch


def peek_array(a):
    return a[:3, :3]


class Timer(object):
    def __init__(self, name="Name", decimals=1):
        self.name = name
        self.decimals = decimals

    def __enter__(self):
        print("Start timer:", self.name)
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        duration = time.time() - self.start
        duration = round(duration, self.decimals)
        print("Ended timer: {}: {} s".format(self.name, duration))


def rbf_for_loop(X, T, gamma=1):
    with Timer("For loop"):
        n = len(X)
        p = len(T)
        rbf = np.zeros(shape=(n, p))
        for i in range(n):
            for j in range(p):
                top = np.square(np.linalg.norm(X[i, :] - T[j, :]))
                rbf[i, j] = np.exp(np.divide(-top, gamma))
        return rbf


def rbf_broadcast_np(X, T, gamma=1):
    with Timer("Numpy broadcasting"):
        n, d1 = X.shape
        p, d2 = T.shape
        assert d1 == d2
        X = np.reshape(X, (n, 1, d1))
        T = np.reshape(T, (1, p, d2))
        rbf = np.subtract(X, T)
        rbf = np.square(rbf)
        rbf = np.sum(rbf, axis=-1)
        rbf = np.exp(np.divide(-rbf, gamma))
        return rbf


def get_norm_square(X, T):
    n, d1 = X.shape
    p, d2 = T.shape
    assert d1 == d2
    X = X.view(n, 1, d1)
    T = T.view(1, p, d2)
    norm = torch.sub(X, T)
    norm = torch.pow(norm, exponent=2)
    norm = torch.sum(norm, dim=-1)
    return norm


def rbf_broadcast_torch(X, T, gamma=1, use_gpu=False):
    with Timer("Pytorch broadcasting"):
        X = torch.Tensor(X)
        T = torch.Tensor(T)
        if use_gpu:
            X = X.cuda()
            T = T.cuda()
        norm_square = get_norm_square(X, T)
        rbf = torch.exp(torch.div(-norm_square, gamma))
        return rbf.cpu().numpy()


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    N = 100
    P = 2000
    D = 3000

    X = np.random.rand(N, D) / 10
    T = np.random.rand(P, D) / 10
    print("X:", X.shape, peek_array(X))
    print("T:", T.shape, peek_array(T))

    print(peek_array(rbf_for_loop(X, T)))
    print(peek_array(rbf_broadcast_np(X, T)))
    print(peek_array(rbf_broadcast_torch(X, T)))
    # print(peek_array(rbf_broadcast_torch(X, T, use_gpu=True)))
