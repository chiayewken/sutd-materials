from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def iteratset(obj, components, value):
    print("components", components)
    if not hasattr(obj, components[0]):
        return False
    elif len(components) == 1:
        setattr(obj, components[0], value)
        return True
    else:
        nextobj = getattr(obj, components[0])
        return iteratset(nextobj, components[1:], value)


def setbyname(model, name, value):
    components = name.split(".")
    success = iteratset(model, components, value)
    return success


def invert_normalize(image: torch.Tensor, mean=None, std=None):
    if std is None:
        std = [0.229, 0.224, 0.225]
    if mean is None:
        mean = [0.485, 0.456, 0.406]

    print(image.shape)
    s = torch.tensor(np.asarray(std, dtype=np.float32)).unsqueeze(1).unsqueeze(2)
    m = torch.tensor(np.asarray(mean, dtype=np.float32)).unsqueeze(1).unsqueeze(2)

    res = image * s + m
    return res


def plot_image_and_gradient(
    grad: torch.Tensor, image: torch.Tensor, q=100, path: Path = None
):
    assert grad.shape == image.shape
    n, c, h, w = grad.shape
    assert n == 1
    grad = grad.squeeze(dim=0)
    image = image.squeeze(dim=0)
    fig, axs = plt.subplots(1, 2)

    grad = grad.sum(dim=0).numpy()
    clim = np.percentile(np.abs(grad), q)
    grad = grad / clim
    axs[1].imshow(grad, cmap="seismic", clim=(-1, 1))
    axs[1].axis("off")

    ts = invert_normalize(image)
    a = ts.data.numpy().transpose((1, 2, 0))
    axs[0].imshow(a)
    axs[0].axis("off")

    plt.show()
    if path is not None:
        plt.savefig(str(path))
