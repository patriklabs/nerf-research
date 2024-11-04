from distutils.util import strtobool

import torch
from torch import nn

from nerf.nerf.nerf_render import NerfRender, integrate_ray

from nerf.nerf.nerf_hypothesis_net import NerfLimit
from nerf.util.util import resample, uniform_sample
from nerf.nerf.kl_div import kl_gauss
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image


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
        homogeneous_projection=True,
        mixtures=4,
        hypo=True,
        low_res_bins=64,
        high_res_bins=128,
        **kwargs
    ) -> None:
        super().__init__()

        self.use_hypo = hypo

        self.render = NerfRender(Lp, Ld, homogeneous_projection)

        if hypo:
            self.bins = bins
            self.nerf_limit = NerfLimit(Lp, homogeneous_projection, k=mixtures)
        else:
            self.render_low_res = NerfRender(Lp, Ld, homogeneous_projection)
            self.low_res_bins = low_res_bins
            self.high_res_bins = high_res_bins

    def forward(self, rays, tn, tf, step):

        if self.use_hypo:
            return self.hypo(rays, tn, tf, step)
        else:
            return self.baseline(rays, tn, tf, step)

    def baseline(self, rays, tn, tf, step):

        tn = torch.zeros_like(tn)
        tf = 3 * torch.ones_like(tf)

        t_low_res = uniform_sample(tn, tf, self.low_res_bins)

        # do one round to find out important sampling regions

        color_low_res, _, w_low_res, t_low_res = self.render_low_res.forward(
            rays, t_low_res
        )

        with torch.no_grad():

            # sample according to w
            t_resamp = resample(w_low_res, t_low_res, self.high_res_bins)

        t_resamp = torch.cat((t_low_res, t_resamp), dim=1)

        color_high_res, depth, w, t = self.render.forward(rays, t_resamp)

        results = {
            "color_high_res": color_high_res,
            "depth": depth,
            "color_low_res": color_low_res,
        }

        if step % 100 == 0 and self.training:

            plot = self.plot_resamp(t_low_res, w_low_res, t, w)

            results.update({"plot": plot})

        return results

    def iterative_resmp(self, rays, tn, tf, step):

        tn = torch.zeros_like(tn)
        tf = 3 * torch.ones_like(tf)

        t = uniform_sample(tn, tf, 32)

        sigma, color = self.render.evaluate_ray(rays, t)

        for _ in range(4):

            with torch.no_grad():

                (
                    _,
                    _,
                    w,
                    _,
                ) = integrate_ray(t, sigma, color)

                t_resamp = resample(w, t, 16)

            sigma_i, color_i = self.render.evaluate_ray(rays, t_resamp)

            t = torch.cat((t, t_resamp), dim=-2)
            sigma = torch.cat((sigma, sigma_i), dim=-2)
            color = torch.cat((color, color_i), dim=-2)

            t, idx = torch.sort(t, dim=-2)

            sigma = torch.gather(sigma, -2, idx)
            color = torch.gather(color, -2, idx.expand(-1, -1, 3))

        color_high_res, depth, w, t = integrate_ray(t, sigma, color)

        results = {"color_high_res": color_high_res, "depth": depth}

        if step % 100 == 0 and self.training:

            plot = self.plot_resamp(t, w, t, w)

            results.update({"plot": plot})

        return results

    def hypo(self, rays, tn, tf, step):

        tn = torch.zeros_like(tn)
        tf = 3 * torch.ones_like(tf)

        t = uniform_sample(tn, tf, self.bins)

        pi, mu, std = self.nerf_limit(rays)

        t_g = mu + 2.0 * std * torch.randn(
            [*t.shape[0:-1], pi.shape[-1]], device=pi.device, dtype=pi.dtype
        )

        u = torch.rand_like(t)

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

    def plot_resamp(self, t, w, t_smp, w_smp):

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
