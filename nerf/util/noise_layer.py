import torch

from torch import nn


class Noise(nn.Module):

    def __init__(self, std) -> None:
        super().__init__()
        self.std = std

    def forward(self, x):

        if self.training:
            x = x + self.std * torch.randn_like(x)

        return x
