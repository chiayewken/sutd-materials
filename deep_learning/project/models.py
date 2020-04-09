import torch


class ConvClassifier(torch.nn.Module):
    """Simple convolutional classifier"""

    def __init__(self, num_in, num_out, num_hidden=64):
        super().__init__()
        self.body = torch.nn.Sequential(
            self.get_block(num_in, num_hidden),
            self.get_block(num_hidden, num_hidden),
            self.get_block(num_hidden, num_hidden),
            self.get_block(num_hidden, num_hidden),
        )
        self.classifier = torch.nn.Linear(num_hidden, num_out)

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

    def __init__(self, num_in, num_out, num_hidden=64, num_layers=3):
        super().__init__()
        layers = [torch.nn.Linear(num_in, num_hidden)]
        for _ in range(num_layers - 2):
            layers.append(torch.nn.Linear(num_hidden, num_hidden))
            layers.append(torch.nn.ReLU())
        self.body = torch.nn.Sequential(*layers)
        self.classifier = torch.nn.Linear(num_hidden, num_out)

    def forward(self, x):
        x = self.body(x)
        x = self.classifier(x)
        return x


class LinearClassifierWithOOS(LinearClassifier):
    def __init__(self, num_in, num_out, num_hidden=64, num_layers=3):
        super().__init__(num_in, num_out, num_hidden, num_layers)
        self.classifier_oos = torch.nn.Linear(num_hidden, 1)

    def forward(self, x):
        x_body = self.body(x)
        x = self.classifier(x_body)
        x_oos = self.classifier_oos(x_body)
        return x, x_oos
