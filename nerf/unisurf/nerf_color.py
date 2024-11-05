import torch
from torch import nn
from torch.nn.utils import weight_norm

from nerf.util.embedding import Embedding


class NerfColor(nn.Module):

    def __init__(self, Ld) -> None:
        super().__init__()

        self.embedding = Embedding(Ld)

        dir_dim = 3

        self.color = nn.Sequential(
            weight_norm(nn.Linear(2 * dir_dim * Ld + dir_dim + 3 + 4 + 256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 3)),
            nn.Sigmoid(),
        )

    def forward(self, x, n, h, d):

        _, N, _ = x.shape

        d = self.embedding(d).expand(-1, N, -1)

        # project onto sphere in p4
        x = torch.cat((x, torch.ones_like(x[..., 0:1])), dim=-1)

        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)

        x = torch.cat((x, n, h, d), dim=-1)

        color = self.color(x)

        return color
