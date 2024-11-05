import torch
from torch import nn

from nerf.util.embedding import Embedding
from nerf.util.noise_layer import Noise


class NerfLimit(nn.Module):

    def __init__(self, Lp, homogeneous_projection=False, k=2) -> None:
        super().__init__()

        self.k = k

        pos_dim = 3

        self.embedding_p = Embedding(Lp, homogeneous_projection)

        Ld = Lp // 2

        self.embedding_d = Embedding(Ld)

        if homogeneous_projection:
            pos_dim += 1

        self.dnn1 = nn.Sequential(
            nn.Linear(2 * pos_dim * Lp + pos_dim + 2 * 3 * Ld + 3, 256),
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

        self.limit = nn.Sequential(nn.Linear(256, 3 * k))

        self.softmax = nn.Softmax(-1)
        self.softplus = nn.Softplus(1)

    def forward(self, ray):

        o, d = torch.split(ray.unsqueeze(-2), [3, 3], dim=-1)

        o = self.embedding_p(o)
        d = self.embedding_d(d)

        x = torch.cat((o, d), dim=-1)

        F = self.dnn1(x)

        sigma = self.limit(F)

        pi, mu, std = sigma.split([self.k, self.k, self.k], dim=-1)

        pi = self.softmax(pi)
        mu = self.softplus(mu)
        std = self.softplus(std)

        return pi, mu, std
