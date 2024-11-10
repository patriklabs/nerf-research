import numpy as np


def reject_outliers(data, m=4):
    """
    Reject outliers.

    Args:
        data (np.ndarray): Data.
        m (int): Multiplier.

    Returns:
        np.ndarray: Data.
    """

    dev = abs(data - np.median(data))
    mdev = np.median(dev)

    return data[dev < m * mdev]
