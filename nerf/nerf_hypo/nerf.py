import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from .kl_div import kl_gauss
from .nerf_hypothesis_net import NerfLimit
from nerf.nerf.nerf_render import NerfRender
from nerf.util.boundary import RayBoundaryUnitSphere
from nerf.util.util import uniform_sample


def draw_categorical(pi, u):

    pi = torch.cumsum(pi, dim=-1)

    cat = torch.argmax((pi > u).float(), dim=-1, keepdim=True)

    return cat


class Nerf(nn.Module):
    def __init__(
        self,
        Lp=10,
        Ld=4,
        bins=32,
        bins_hypo=128,
        homogeneous_projection=True,
        mixtures=4,
        ray_boundary=RayBoundaryUnitSphere(),
        **kwargs
    ) -> None:
        super().__init__()
        self.bins = bins
        self.bins_hypo = bins_hypo
        self.render = NerfRender(Lp, Ld, homogeneous_projection)
        self.nerf_limit = NerfLimit(Lp, homogeneous_projection, k=mixtures)
        self.ray_boundary = ray_boundary

    def forward(self, rays, tn, tf, step):

        tn, tf = self.ray_boundary.boundary(rays, tn, tf)

        t = uniform_sample(tn, tf, self.bins)

        pi, mu, std = self.nerf_limit(rays)

        t_g = mu + std * torch.randn(
            [t.shape[0], self.bins_hypo, pi.shape[-1]],
            device=pi.device,
            dtype=pi.dtype,
        )

        u = torch.rand(
            [t.shape[0], self.bins_hypo, 1],
            dtype=t.dtype,
            layout=t.layout,
            device=t.device,
        )

        cat = draw_categorical(pi, u)

        t_g = torch.gather(t_g, dim=-1, index=cat)

        t = torch.cat((t, t_g.clamp_min(tn.unsqueeze(-1)).detach()), dim=1)

        color_high_res, depth, w, t = self.render.forward(rays, t)

        div = kl_gauss(t.detach(), w.detach(), pi, mu, std)

        results = {
            "color_high_res": color_high_res,
            "depth": depth,
            "reg_val": 1e-2 * div,
        }

        if step % 100 == 0 and self.training:

            plot = self.plot(t, w, pi, mu, std)

            results.update({"plot": plot})

        return results

    def plot(self, t, w, pi, mu, std, t_resamp=None):

        q = pi * torch.exp(-0.5 * ((t - mu) / (std)) ** 2) / ((std) * 2.50662827463)
        q = q.sum(-1, keepdim=True)

        q = q[-1].reshape(-1).detach().cpu().numpy()

        t_smp = None

        if t_resamp is not None:
            t_smp = t_resamp[-1].reshape(-1).detach().cpu().numpy()

        w = w[-1].reshape(-1).detach().cpu().numpy()
        t = t[-1].reshape(-1).detach().cpu().numpy()

        t_max = int(np.ceil(np.max(t)))

        plots = 4

        if t_smp is not None:
            plots += 2

        count = 0

        fig, axs = plt.subplots(plots)

        fig.suptitle("sampling")

        if t_smp is not None:
            axs[count].hist(t_smp)
            axs[count].set_xlim([0, t_max])
            axs[count].set_title("t smp hist")
            count += 1

        axs[count].plot(t, w)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("ray weights")
        count += 1

        if t_smp is not None:

            axs[count].plot(t_smp, np.zeros_like(t_smp), "*")
            axs[count].set_xlim([0, t_max])
            axs[count].set_title("t smp")
            count += 1

        axs[count].hist(t)
        axs[count].set_xlim([0, t_max])
        count += 1

        axs[count].plot(t, np.zeros_like(t), "*")
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("t")
        count += 1

        axs[count].plot(t, q)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("q dist")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        plt.close()

        return (
            torch.tensor(np.array(Image.open(buf)))
            .permute(2, 0, 1)
            .unsqueeze(0)[:, 0:3]
        )
