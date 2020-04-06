from typing import Tuple

import torch

from utils import HyperParams


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(TemporalConvNet):
    """
    Wrapper as drop-in replacement for LSTM/GRU modules
    Adapted from https://github.com/locuslab/TCN
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        batch_first=False,
        dropout=0.0,
        kernel_size=2,
    ):
        super().__init__(
            num_inputs=input_size,
            num_channels=[hidden_size] * num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.batch_first = batch_first

    def forward(
        self, x: torch.Tensor, dummy_input=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1)
        # Convolution convention is channels first so swap
        x = super().forward(x.transpose(1, 2)).transpose(1, 2)
        return x, dummy_input


class SequenceNet(torch.nn.Module):
    def __init__(self, n_vocab: int, hparams: HyperParams, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.embed = torch.nn.Embedding(n_vocab, hparams.n_hidden)
        selector = dict(lstm=torch.nn.LSTM, gru=torch.nn.GRU, tcn=TCN)
        self.net = selector[hparams.model](
            input_size=hparams.n_hidden,
            hidden_size=hparams.n_hidden,
            num_layers=hparams.n_layers,
            batch_first=self.batch_first,
            dropout=hparams.dropout,
        )
        self.linear = torch.nn.Linear(hparams.n_hidden, n_vocab)
        if hparams.tie_embed_weights:
            self.linear.weight = self.embed.weight

    def forward(
        self, x: torch.Tensor, states: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(x)
        x, states = self.net(x, states)
        x = self.linear(x)
        return x, states


def main():
    num_hidden = 128
    num_layers = 3
    batch_size = 4
    sequence_length = 12

    layer = TCN(
        input_size=num_hidden,
        hidden_size=num_hidden,
        num_layers=num_layers,
        batch_first=True,
    )
    inputs = torch.zeros(batch_size, sequence_length, num_hidden)
    outputs = layer(inputs)
    assert outputs.shape == (batch_size, sequence_length, num_hidden)


if __name__ == "__main__":
    main()
