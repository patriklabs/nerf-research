import context
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.colmap_solution import ColmapSolution
from nerf.util.util import resample, uniform_sample, resample, resample_new
from nerf.nerf_neus.nerf_render import Alpha


def test_alpha():

    alpha = Alpha()

    for _ in range(1024):

        x1 = torch.randn((1, 1, 1))
        x2 = torch.randn((1, 1, 1))

        y1 = alpha.forward(x1, x2)

        y2 = alpha.test(x1, x2)

        diff = (y1 - y2).abs()

        diff_max = diff.max()

        if not torch.isnan(y2):

            assert diff_max < 1e-6


def test_alpha2():

    alpha = Alpha()

    x1 = torch.randn((1024, 64, 1))
    x2 = torch.randn((1024, 64, 1))

    y1 = alpha.forward(x1, x2)

    y2 = alpha.test(x1, x2)

    diff = (y1 - y2).abs()

    diff_max = diff.max()

    assert (y1 <= 1.0).all()
    assert (y2 <= 1.0).all()

    assert (y1 >= 0.0).all()
    assert (y2 >= 0.0).all()

    assert diff_max < 1e-6


def test_net_gen():

    test = ColmapSolution("/database/colmap_test", 0, [320, 320])

    test.unit_rescale()

    rays = test.calculate_rays()

    print(rays)


def test_resample_new():

    w_smp = 32
    t = torch.linspace(0, 20, w_smp).reshape(1, w_smp, 1)

    mean = 12.0
    std = 2.0

    x_min = mean - 4 * std
    x_max = mean + 4 * std

    w = torch.exp(-((t - mean) / std).pow(2))

    t_smp = resample_new(w, t, 1000)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    mean_test = t_smp.mean(1)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test")
    axs[0].hist(t_smp)
    axs[0].set_xlim([x_min, x_max])
    axs[1].plot(t, w)
    axs[1].set_xlim([x_min, x_max])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([x_min, x_max])

    plt.show()


def test_resample():

    t = torch.linspace(0, 20, 64).reshape(1, 64, 1)

    mean = 12.0
    std = 2.0

    x_min = mean - 4 * std
    x_max = mean + 4 * std

    w = torch.exp(-((t - mean) / std).pow(2))

    t_smp = resample(w, t, 10000)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    mean_test = t_smp.mean(1)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test")
    axs[0].hist(t_smp)
    axs[0].set_xlim([x_min, x_max])
    axs[1].plot(t, w)
    axs[1].set_xlim([x_min, x_max])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([x_min, x_max])

    plt.show()


def test_resample2():

    t = torch.linspace(0, 20, 1024).reshape(1, 1024, 1)

    mean = [4, 7, 12, 18]
    std = [0.1, 0.2, 0.1, 0.3]
    pi = [1, 3, 2, 0.5]
    pi_sum = sum(pi)
    pi = [val / pi_sum for val in pi]

    w = [torch.exp(-((t - mean) / std).pow(2)) for mean, std in zip(mean, std)]

    w = torch.stack([pi * w for pi, w in zip(pi, w)]).mean(0)

    t_smp = resample(w, t, 1024)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test")
    axs[0].hist(t_smp)
    axs[0].set_xlim([0, 20])
    axs[1].plot(t, w)
    axs[1].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[1].set_xlim([0, 20])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([0, 20])

    plt.show()


def test_resample():

    t = torch.linspace(0, 20, 64).reshape(1, 64, 1)

    mean = 12.0
    std = 2.0

    x_min = mean - 4 * std
    x_max = mean + 4 * std

    w = torch.exp(-((t - mean) / std).pow(2))

    t_smp = resample(w, t, 10000)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    mean_test = t_smp.mean(1)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle("test")
    axs[0].hist(t_smp)
    axs[0].set_xlim([x_min, x_max])
    axs[1].plot(t, w)
    axs[1].set_xlim([x_min, x_max])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([x_min, x_max])

    plt.show()


def test_uniform():

    tn = torch.tensor([5]).view(1, 1).expand(1024, 1).float()

    tf = torch.tensor([20]).expand(1024, 1).float()

    t = uniform_sample(tn, tf, 64)

    x = torch.tensor(list(range(64))).view(1, 64, 1).expand(1024, 64, 1).float()

    t = t.reshape(-1).numpy()
    x = x.reshape(-1).numpy()

    print("")

    plt.scatter(x, t)
    plt.show()
