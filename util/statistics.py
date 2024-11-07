import numpy as np


def reject_outliers(data, m=4):

    dev = abs(data - np.median(data))
    mdev = np.median(dev)

    return data[dev < m * mdev]
