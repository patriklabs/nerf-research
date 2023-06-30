import torch
from torch import nn
from torch.nn.utils import weight_norm

from nerf.util.embedding import Embedding

class NerfDensity(nn.Module):

    def __init__(self, Lp, homogeneous_projection=False) -> None:
        super().__init__()

        pos_dim = 3

        self.embedding = Embedding(Lp, homogeneous_projection)

        if homogeneous_projection:
            pos_dim += 1

        self.dnn1 = nn.Sequential(
            weight_norm(nn.Linear(2*pos_dim*Lp+pos_dim, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100)
        )

        self.dnn2 = nn.Sequential(
            weight_norm(nn.Linear(2*pos_dim*Lp+256+pos_dim, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100),
            weight_norm(nn.Linear(256, 256)),
            nn.Softplus(beta=100)
        )

        self.density = nn.Sequential(
            weight_norm(nn.Linear(256, 1))
        )

    def forward(self, x):

        x = self.embedding(x)

        F = self.dnn1(x)

        F = torch.cat((F, x), dim=-1)

        F = self.dnn2(F)

        sigma = self.density(F)

        return sigma, F
