import torch

from torch import nn
import math


class Embedding(nn.Module):

    def __init__(self, L, homogeneous_projection=False) -> None:
        super().__init__()

        self.L = L

        self.homogeneous_projection = homogeneous_projection

    def forward(self, x):

        if self.homogeneous_projection:

            x = torch.cat((x, torch.ones_like(x[..., 0:1])), dim=-1)

            x = x / torch.linalg.norm(x, dim=-1, keepdim=True)

        x_e = torch.cat(
            [
                torch.cat(
                    [torch.sin((2**i) * math.pi * x), torch.cos((2**i) * math.pi * x)],
                    dim=-1,
                )
                for i in range(self.L)
            ],
            dim=-1,
        )

        return torch.cat((x, x_e), dim=-1)
