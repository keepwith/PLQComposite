import numpy as np
from scipy.special import huber


def relu(x):
    """
    :param x:
    :return:
    """
    return np.maximum(x, 0)


def rehu(x, cut=1):
    """

    :param x:
    :param cut:
    :return:
    """
    n_samples = x.shape[0]
    cut = cut * np.ones_like(x)

    u = np.maximum(x, 0)
    return huber(cut, u)
