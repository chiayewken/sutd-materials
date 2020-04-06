import torch
import torch.nn as nn
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.randn(1, self.hidden_size)


class MyLSTM(RNN):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.i2h = None
        self.i2o = None
        self.net = torch.nn.LSTM(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input: torch.Tensor, hidden) -> (torch.Tensor, tuple):
        if input.ndim == 2:
            input = input.unsqueeze(dim=0)

        output, hidden = self.net(input, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return (
            torch.randn(1, 1, self.net.hidden_size),
            torch.randn(1, 1, self.net.hidden_size),
        )


class MyGRU(MyLSTM):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.net = torch.nn.GRU(input_size, hidden_size)
