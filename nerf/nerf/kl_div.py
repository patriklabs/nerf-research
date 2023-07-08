import torch
from nerf.util.util import where


def kl_gauss(t, w, pi, mu, std):

    std = std.clamp_min(1e-6)

    q = pi*torch.exp(-0.5*((t-mu)/(std))**2)/((std)*2.50662827463)

    q = q.sum(-1, keepdim=True)

    div = -w*torch.log(q.clamp_min(1e-6))

    return div.mean(1)


def kl_div(t, w, tn, tf):

    b = 1/(tf-tn)
    b = b.unsqueeze(-1)

    t1 = t[:, :-1]
    t2 = t[:, 1:]

    w1 = w[:, :-1]
    w2 = w[:, 1:]

    delta_t = (t2-t1)

    k = (w2-w1)/where(delta_t > 0, delta_t, torch.ones_like(delta_t))
    m = w1 - k*t1

    c1 = 0.5*k*(t1**2) + m*t1
    c2 = 0.5*k*(t2**2) + m*t2

    C = (c2-c1).sum(dim=1, keepdim=True)

    C = where(C > 0, C, torch.ones_like(C))

    w = w/C

    w_hat = where(w > 0, w, torch.ones_like(w))

    div = w*torch.log(w_hat/b)

    return div.mean(1)/b
