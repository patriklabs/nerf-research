import torch
from torch import nn

from nerf.util.embedding import Embedding


class NerfColor(nn.Module):

    def __init__(self, Ld) -> None:
        super().__init__()

        self.embedding = Embedding(Ld)

        self.color = nn.Sequential(
            nn.Linear(256 + 2 * 3 * Ld + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, F, d):

        _, N, _ = F.shape

        d = self.embedding(d).expand(-1, N, -1)

        F = torch.cat((F, d), dim=-1)

        color = self.color(F)

        return color
