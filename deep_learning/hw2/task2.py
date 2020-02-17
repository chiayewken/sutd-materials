import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import decomposition


def peek_array(a):
    return a[:3, :3]


def get_blobs(_n_blobs, _n_per_blob, _n_dim):
    means = np.random.randn(_n_blobs, _n_dim)
    stds = np.random.randn(_n_blobs, _n_dim)
    return [
        torch.randn(_n_per_blob, _n_dim) * stds[i] + means[i] for i in range(n_blobs)
    ]


def make_x(blobs):
    x = torch.cat(blobs)
    idxs = np.arange(len(x))
    np.random.shuffle(idxs)
    x = x[idxs]
    print("x:", x.shape)
    return x


def get_dists(_X, _T):
    n, d1 = _X.shape
    p, d2 = _T.shape
    assert d1 == d2
    _X = _X.view(n, 1, d1)
    _T = _T.view(1, p, d2)
    dists = torch.sub(_X, _T)
    dists = torch.pow(dists, exponent=2)
    dists = torch.sum(dists, dim=-1)
    return dists


def plot_xs(xs, pca, title=""):
    for x in xs:
        x = x.numpy()
        x = pca.transform(x)
        plt.scatter(*zip(*x))
    plt.title(title)
    plt.show()


def check_threshold(T, T_old, threshold):
    dists = get_dists(T, T_old)
    dists = torch.diagonal(dists)
    dists = dists.numpy()
    return all([(d < threshold) for d in dists])


def plot_T(T, title, pca):
    T = T.numpy()
    T = pca.transform(T)
    plt.scatter(*zip(*T))
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    n_clusters = 5
    n_iters = 10
    n_blobs = 5
    n_per_blob = 50
    n_dim = 20
    n_data = n_blobs * n_per_blob
    threshold = 1e-3

    blobs = get_blobs(n_blobs, n_per_blob, n_dim)
    X = make_x(blobs)
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    plot_xs(blobs, pca, title="Original blobs")
    print(peek_array(X.numpy()))
    T = torch.clone(X[:n_clusters])  # First n, random since X shuffled
    idxs_nearest_centroid = None
    plot_T(T, "Initial cluster centers", pca)

    for m in range(n_iters):
        T_old = torch.clone(T)
        dists = get_dists(X, T)
        assert dists.shape == (n_data, n_clusters), dists.shape
        print("{} Loss:".format(m), torch.mean(torch.min(dists, dim=-1)[0]).numpy())
        idxs_nearest_centroid = torch.argmin(dists, dim=-1)

        for i in range(n_clusters):
            mask = idxs_nearest_centroid == i
            members = X[mask]
            if len(members) > 0:
                T[i] = torch.mean(members, dim=0)

        if check_threshold(T, T_old, threshold):
            break

    plot_T(T, "Converged cluster centers", pca)
    blobs_pred = [X[idxs_nearest_centroid == i] for i in range(n_clusters)]
    plot_xs(blobs_pred, pca, title="Predicted blobs")
