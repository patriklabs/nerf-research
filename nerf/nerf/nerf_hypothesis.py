
from distutils.util import strtobool

import torch
from torch import nn

from nerf.nerf.nerf_render import NerfRender

from nerf.nerf.nerf_hypothesis_net import NerfLimit
from nerf.util.util import resample, uniform_sample
from nerf.nerf.kl_div import kl_gauss
import matplotlib.pyplot as plt
import numpy as np


def draw_categorical(pi, u):

    pi = torch.cumsum(pi, dim=-1)

    cat = torch.argmax((pi > u).float(), dim=-1, keepdim=True)

    return cat


class Nerf(nn.Module):
    def __init__(self, Lp, Ld, bins, homogeneous_projection, mixtures, **kwargs) -> None:
        super().__init__()

        self.bins = bins

        self.render = NerfRender(Lp, Ld, homogeneous_projection)
        self.nerf_limit = NerfLimit(Lp, homogeneous_projection, k=mixtures)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Nerf")

        parser.add_argument("--Lp", type=int, default=10)
        parser.add_argument("--Ld", type=int, default=4)
        parser.add_argument("--bins", type=int, default=32)
        parser.add_argument("--mixtures", type=int, default=4)

        parser.add_argument("--homogeneous_projection",
                            type=strtobool, default=True)

        return parent_parser

    def forward(self, rays, tn, tf, step):

        tn = torch.zeros_like(tn)
        tf = 3*torch.ones_like(tf)

        t = uniform_sample(tn, tf, self.bins)

        pi, mu, std = self.nerf_limit(rays)

        t_g = mu + std * \
            torch.randn([*t.shape[0:-1], pi.shape[-1]],
                        device=pi.device, dtype=pi.dtype)

        u = torch.rand_like(t)

        cat = draw_categorical(pi, u)

        t_g = torch.gather(t_g, dim=-1, index=cat)

        t = torch.cat((t, t_g.clamp_min(tn.unsqueeze(-1)).detach()), dim=1)

        color_high_res, depth, w, t = self.render.forward(rays, t)

        div = kl_gauss(t.detach(), w.detach(), pi, mu, std)

        if False:  # step % 1000 == 0 and self.training:

            q = pi*torch.exp(-0.5*((t-mu)/(std)) ** 2)/((std)*2.50662827463)
            q = q.sum(-1, keepdim=True)

            w = w[-1].reshape(-1).detach().cpu().numpy()
            t = t[-1].reshape(-1).detach().cpu().numpy()
            q = q[-1].reshape(-1).detach().cpu().numpy()

            w_sum = np.sum(w)

            fig, axs = plt.subplots(2)
            fig.suptitle('dist')
            axs[0].title.set_text('w')
            axs[0].plot(t, w)
            axs[0].set_xlim([0, 1])
            axs[1].title.set_text('gauss')
            axs[1].plot(t, q)
            axs[1].set_xlim([0, 1])

            plt.show()

        return {"color_high_res": color_high_res, "depth": depth, "curv": 1e-2*div}
