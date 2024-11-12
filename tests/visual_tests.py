import context
import matplotlib.pyplot as plt
import numpy as np
import torch

from nerf.util.util import resample


def visualize_resample(w_smp=32):

    t = torch.linspace(0, 20, w_smp).reshape(1, w_smp, 1)

    mean = 12.0
    std = 2.0

    x_min = mean - 4 * std
    x_max = mean + 4 * std

    w = torch.exp(-0.5 * ((t - mean) / std).pow(2)) / (
        std * torch.sqrt(2 * torch.tensor(np.pi))
    )

    t_smp = resample(w, t, 1000)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test normal distribution")
    axs[0].hist(t_smp)
    axs[0].set_xlim([x_min, x_max])
    axs[1].plot(t, w)
    axs[1].set_xlim([x_min, x_max])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([x_min, x_max])

    plt.show()


def visualize_mixed_normal_distributions():

    t = torch.linspace(0, 20, 1024).reshape(1, 1024, 1)

    mean = [4, 7, 12, 18]
    std = [0.1, 0.2, 0.1, 0.3]
    pi = [1, 3, 2, 0.5]
    pi_sum = sum(pi)
    pi = [val / pi_sum for val in pi]

    w = [
        torch.exp(-0.5 * ((t - mean) / std).pow(2))
        / (std * torch.sqrt(2 * torch.tensor(np.pi)))
        for mean, std in zip(mean, std)
    ]

    w = torch.stack([pi * w for pi, w in zip(pi, w)]).mean(0)

    t_smp = resample(w, t, 1024)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test - mixed normal distributions")
    axs[0].hist(t_smp, bins=100)
    axs[0].set_xlim([0, 20])
    axs[1].plot(t, w)
    axs[1].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[1].set_xlim([0, 20])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([0, 20])

    plt.show()


if __name__ == "__main__":
    visualize_resample()
    visualize_mixed_normal_distributions()
