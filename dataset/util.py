import numpy as np


def generate_grid(H, W):

    x = np.linspace(0, W - 1, W)

    y = np.linspace(0, H - 1, H)

    xv, yv = np.meshgrid(x, y)

    return np.stack((xv, yv), axis=0)


def to_homogen(x):

    return np.concatenate((x, np.ones_like(x[0:1])), axis=0)


def to_K_matrix(intrinsics):

    K = np.eye(4, 4)

    fx, fy, cx, cy = intrinsics

    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    return K


def iqr(x):

    q3, q1 = np.percentile(x, [75, 25], axis=-1)

    iqr = q3 - q1

    L = q1 - 1.5 * iqr
    U = q3 + 1.5 * iqr

    return L, U
