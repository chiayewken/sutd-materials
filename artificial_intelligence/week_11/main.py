"""
Parametrized hooks, backward hooks
Class with custom backward method, wrapped autograd, network attribution

Dataset return single image from 500 imagenet validation set


"""
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets.utils import extract_archive, download_and_extract_archive
from torchvision.models import vgg16, vgg16_bn, VGG
from tqdm import tqdm

from guidedbpcodehelpers import plot_image_and_gradient
from imgnetdatastuff import dataset_imagenetvalpart


class Sample(BaseModel):
    image: torch.Tensor
    label: int
    filename: str

    class Config:
        arbitrary_types_allowed = True


class ImageNetValDataset(dataset_imagenetvalpart):
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.root = root
        self.download()
        root_images, root_labels = self.extract()
        super().__init__(
            root_dir=str(root_images),
            xmllabeldir=str(root_labels),
            synsetfile=str(root / "synset_words.txt"),
            maxnum=500,
            transform=transform,
        )
        print({self.__class__.__name__: len(self)})

    def download(self):
        url = "https://github.com/chiayewken/sutd-materials/releases/download/v0.1.0/sutd_ai_week_11_homework.zip"
        download_and_extract_archive(
            url, download_root="/tmp", extract_root=str(self.root)
        )

    def extract(self):
        zipped = self.root / "imgnet500.zip"
        root_images = self.root / "images"
        if not root_images.exists():
            extract_archive(str(zipped), str(root_images))
        root_images = root_images / "mnt/scratch1/data/imagespart"
        assert root_images.exists()
        print(dict(root_images=list(root_images.iterdir())[:10]))

        compressed = self.root / "ILSVRC2012_bbox_val_v3.tgz"
        root_labels = self.root / "labels"
        if not root_labels.exists():
            extract_archive(str(compressed), str(root_labels))
        root_labels = root_labels / "val"
        assert root_labels.exists()
        print(dict(root_labels=list(root_labels.iterdir())[:10]))

        return root_images, root_labels

    def convert_to_onehot(self, i: int) -> torch.Tensor:
        num_classes = len(self.clsdict)
        onehot = torch.zeros(num_classes)
        onehot[i] = 1.0
        return onehot


def get_transform():
    # Reference: https://pytorch.org/hub/pytorch_vision_vgg/
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_image_channel_norm(image: torch.Tensor) -> torch.Tensor:
    n, c, h, w = image.shape
    assert n == 1
    x = image.view(c, h * w)
    x = torch.square(x)
    x = torch.sum(x, dim=1)
    x = torch.sqrt(x)
    assert list(x.shape) == [c]
    return x


class Hook:
    def __init__(self, module: torch.nn.Module):
        self.handle = self.register(module)

    def hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
    ):
        raise NotImplementedError

    def register(self, module: torch.nn.Module):
        return module.register_backward_hook(self.hook)

    def remove(self):
        self.handle.remove()


class GradNormHook(Hook):
    def __init__(self, net: VGG, root: Path, filename: str, i_layer: int):
        self.root = root
        self.filename = filename
        self.i_layer = i_layer
        self.value = torch.empty(1)
        super().__init__(module=net.features[i_layer])

    def hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
    ):
        x = outputs[0]
        norm = get_image_channel_norm(x)
        # info = dict(
        #     module=type(module).__name__,
        #     inputs=[type(x) for x in inputs],
        #     outputs=[type(x) for x in outputs],
        #     x=x.shape,
        # )
        # print(info)
        self.value = norm
        stem = Path(self.filename).stem
        name = f"{stem}_conv_{self.i_layer}"
        path = (self.root / name).with_suffix(".npy")
        print(dict(path=path))
        np.save(str(path), norm.cpu().numpy())


class GuidedBackPropHook(Hook):
    def hook(
        self,
        module: torch.nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
    ):
        if isinstance(module, torch.nn.ReLU):
            inputs = inputs[0].clone()
            inputs[inputs < 0] = 0
            return tuple([inputs])


def get_random_samples(dataset: ImageNetValDataset, num: int, seed=42) -> List[Sample]:
    np.random.seed(seed)
    assert num <= len(dataset)
    indices = np.random.choice(len(dataset), num, replace=False)
    return [Sample(**dataset[i]) for i in indices]


def compute_accuracy(samples: List[Sample], net: torch.nn.Module) -> float:
    is_correct = []
    net.eval()

    with torch.no_grad():
        for s in tqdm(samples):
            inputs = s.image.unsqueeze(dim=0)
            outputs: torch.Tensor = net(inputs)
            prediction = outputs.squeeze(dim=0).argmax().item()
            assert isinstance(prediction, int)
            is_correct.append(prediction == s.label)

    score = sum(is_correct) / len(is_correct)
    assert 0.0 <= score <= 1.0
    return score


def run_single_sample(sample: Sample, net: torch.nn.Module):
    loss_fn = torch.nn.CrossEntropyLoss()
    device = list(net.parameters())[0].device
    inputs = Variable(sample.image.unsqueeze(dim=0), requires_grad=True)
    inputs = inputs.to(device)

    outputs = net(inputs)
    targets = torch.from_numpy(np.array([sample.label])).long()
    loss = loss_fn(outputs, targets)
    print(dict(loss=loss))
    loss.backward()
    return inputs.cpu()


def setup_folder(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir()
    return path


def visualize_gradient(inputs: torch.Tensor, filename: str, root: Path):
    stem = Path(filename).stem
    path = (root / stem).with_suffix(".jpg")
    plot_image_and_gradient(inputs.grad.data, inputs, path=path)


def get_device() -> torch.device:
    return torch.device("cpu" if torch.cuda.is_available() else "cuda")


class GBPReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[grad_output < 0] = 0
        return grad_input


class GBPModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return GBPReLU.apply(input)


def apply_guided_backprop(net: torch.nn.Module):
    # for mod in net.modules():
    #     GuidedBackPropHook(mod)

    for name, child in net.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(net, name, GBPModule())
        else:
            apply_guided_backprop(child)


def get_net(use_batch_norm: bool) -> VGG:
    device = get_device()
    net_fn = vgg16_bn if use_batch_norm else vgg16
    net = net_fn(pretrained=True)
    net = net.to(device)
    net = net.eval()
    return net


def analyze_grad_norm(root: Path, samples: List[Sample], use_batch_norm: bool):
    root = root / f"grad_norm_bn_{use_batch_norm}"
    setup_folder(root)
    net = get_net(use_batch_norm)
    print(dict(accuracy=compute_accuracy(samples, net)))

    s: Sample
    for s in tqdm(samples):
        hook = GradNormHook(net, root, s.filename, i_layer=0)
        run_single_sample(s, net)
        hook.remove()

    values = [np.load(str(p)) for p in root.iterdir()]
    stacked = np.stack(values)
    print(dict(grad_norm=dict(mean=stacked.mean(), std=stacked.std())))


def analyze_image_grad(root, samples: List[Sample], use_batch_norm: bool):
    root = root / f"image_grad_bn_{use_batch_norm}"
    setup_folder(root)
    net = get_net(use_batch_norm)
    apply_guided_backprop(net)
    # print(net)

    s: Sample
    for s in tqdm(samples):
        inputs = run_single_sample(s, net)
        visualize_gradient(inputs, s.filename, root)


def main():
    """
    Usage

    1. pip install -r requirements.txt
    2. python main.py
    3. The gradient norm mean and std for with/without batch norm will be printed
    4. All results and images will be saved to "data" folder
    """
    root = Path("data")
    dataset = ImageNetValDataset(root, transform=get_transform())
    samples = get_random_samples(dataset, num=250)

    for use_batch_norm in [True, False]:
        analyze_grad_norm(root, samples, use_batch_norm)
        analyze_image_grad(root, samples[:5], use_batch_norm)


if __name__ == "__main__":
    main()
