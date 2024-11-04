import torch
from torch import nn

from nerf.util.embedding import Embedding
from nerf.util.noise_layer import Noise


class NerfDensity(nn.Module):

    def __init__(self, Lp, homogeneous_projection=False) -> None:
        super().__init__()

        pos_dim = 3

        self.embedding = Embedding(Lp, homogeneous_projection)

        if homogeneous_projection:
            pos_dim += 1

        self.dnn1 = nn.Sequential(
            nn.Linear(2 * pos_dim * Lp + pos_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.dnn2 = nn.Sequential(
            nn.Linear(2 * pos_dim * Lp + pos_dim + 256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.density = nn.Sequential(nn.Linear(256, 1), Noise(1e-2), nn.ReLU())

    def forward(self, x):

        x = self.embedding(x)

        F = self.dnn1(x)

        F = torch.cat((F, x), dim=-1)

        F = self.dnn2(F)

        sigma = self.density(F)

        return sigma, F
