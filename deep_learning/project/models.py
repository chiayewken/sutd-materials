import torch


class ConvClassifier(torch.nn.Module):
    """Simple convolutional classifier"""

    def __init__(self, size_in, size_out, size_hidden=64):
        super().__init__()
        self.body = torch.nn.Sequential(
            self.get_block(size_in, size_hidden),
            self.get_block(size_hidden, size_hidden),
            self.get_block(size_hidden, size_hidden),
            self.get_block(size_hidden, size_hidden),
        )
        self.classifier = torch.nn.Linear(size_hidden, size_out)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    @classmethod
    def get_block(cls, channels_in, channels_out, size_kernel=3):
        pad = cls.get_same_conv_pad(size_kernel)
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, size_kernel, padding=pad),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

    @staticmethod
    def get_same_conv_pad(k):
        """
        Assume stride and dilation are default
        s_out = s_in + 2p - k + 1
        Assume s_out == s_in
        p = (k - 1) / 2
        """
        assert (k - 1) % 2 == 0
        return (k - 1) // 2


class LinearClassifier(torch.nn.Module):
    """Simple multi-linear classifier"""

    def __init__(self, size_in, size_out, size_hidden=64, num_layers=3):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(size_hidden, size_hidden))
            layers.append(torch.nn.ReLU())
        layers[0] = torch.nn.Linear(size_in, size_hidden)
        layers.append(torch.nn.Linear(size_hidden, size_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
