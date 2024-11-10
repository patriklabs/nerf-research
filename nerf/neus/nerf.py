import io

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn

from nerf.neus.nerf_render import NerfRender
from nerf.util.util import resample, uniform_sample


class Nerf(nn.Module):
    def __init__(
        self,
        Lp=10,
        Ld=4,
        homogeneous_projection=True,
        low_res_bins=64,
        high_res_bins=128,
        **kwargs
    ) -> None:
        super().__init__()

        self.render = NerfRender(Lp, Ld, homogeneous_projection)
        self.low_res_bins = low_res_bins
        self.high_res_bins = high_res_bins
        self.s_inv_learned_log = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("s_inv_fixed_log", torch.tensor(-4.0))

    def forward(self, rays, tn, tf, step):

        # tn = torch.zeros_like(tn)
        # tf = 3 * torch.ones_like(tf)

        with torch.no_grad():

            t_low_res = uniform_sample(tn, tf, self.low_res_bins)

            # do one round to find out important sampling regions
            (_, _, w_low_res, t_low_res), _ = self.render.forward(
                rays, t_low_res, self.s_inv_fixed_log, False
            )

            # sample according to w
            t_resamp = resample(w_low_res, t_low_res, self.high_res_bins)

        t_resamp = torch.cat((t_low_res, t_resamp), dim=1)

        (color_high_res, depth, w, t), eikonal_loss = self.render.forward(
            rays, t_resamp, self.s_inv_learned_log
        )

        results = {
            "color_high_res": color_high_res,
            "depth": depth,
            "eikonal_loss": eikonal_loss,
        }

        if step % 100 == 0 and self.training:

            results.update({"plot": self.plot(t_low_res, w_low_res, t, w)})

        return results

    def plot(self, t, w, t_smp, w_smp):

        w = w[-1].reshape(-1).detach().cpu().numpy()
        t = t[-1].reshape(-1).detach().cpu().numpy()

        t_smp = t_smp[-1].reshape(-1).detach().cpu().numpy()
        w_smp = w_smp[-1].reshape(-1).detach().cpu().numpy()

        t_max = int(np.ceil(np.max(t_smp)))

        plots = 6

        fig, axs = plt.subplots(plots)

        fig.suptitle("sampling")

        count = 0
        axs[count].hist(t)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("t low res hist")
        count += 1

        axs[count].hist(t_smp)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("t high res hist")
        count += 1

        axs[count].plot(t, w)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("low res ray weights")
        count += 1

        axs[count].plot(t, np.zeros_like(t), "*")
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("low res smp")
        count += 1

        axs[count].plot(t_smp, w_smp)
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("high res ray weights")
        count += 1

        axs[count].plot(t_smp, np.zeros_like(t_smp), "*")
        axs[count].set_xlim([0, t_max])
        axs[count].set_title("high res smp")
        count += 1

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        plt.close()

        return (
            torch.tensor(np.array(Image.open(buf)))
            .permute(2, 0, 1)
            .unsqueeze(0)[:, 0:3]
        )
