import context
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset.colmap_solution import ColmapSolution
from nerf.util.util import resample, uniform_sample


def test_net_gen():

    test = ColmapSolution("/database/colmap_test", 0, [320, 320])

    test.unit_rescale()

    rays = test.calculate_rays()

    print(rays)


def test_resample():

    t = torch.linspace(0, 20, 64).reshape(1, 64, 1)

    mean = 8.0
    std = 3

    w = torch.exp(-((t-mean)/std).pow(2))

    t_smp = resample(w, t, 128)

    t_smp, _ = torch.sort(t_smp, dim=-2)

    t_smp = t_smp.reshape(-1).numpy()
    w = w.reshape(-1).numpy()
    t = t.reshape(-1).numpy()

    fig, axs = plt.subplots(3)
    fig.suptitle('test')
    axs[0].hist(t_smp)
    axs[0].set_xlim([0, 20])
    axs[1].plot(t, w)
    axs[1].set_xlim([0, 20])
    axs[2].plot(t_smp, np.zeros_like(t_smp), "*")
    axs[2].set_xlim([0, 20])

    plt.show()


def test_uniform():

    tn = torch.tensor([5]).view(1, 1).expand(1024, 1).float()

    tf = torch.tensor([20]).expand(1024, 1).float()

    t = uniform_sample(tn, tf, 64)

    x = torch.tensor(list(range(64))).view(
        1, 64, 1).expand(1024, 64, 1).float()

    t = t.reshape(-1).numpy()
    x = x.reshape(-1).numpy()

    print("")

    plt.scatter(x, t)
    plt.show()
