"""Base functions for the PLQCOM package."""

# Author: Ben Dai <bendai@cuhk.edu.hk>
#         Yixuan Qiu <qiuyixuan@sufe.edu.cn>

# License: MIT License

import numpy as np
from scipy.special import huber


def relu(x):
    """
    Evaluate the ReLU given a vector

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
    Training vector, where `n_samples` is the number of samples

    Returns
    -------
    array of shape (n_samples, )
        An array with ReLU applied, i.e., all negative elements are replaced with 0.

    """
    return np.maximum(x, 0)


def rehu(x, cut=1):
    """
    Evaluate the ReHU given a vector

    Parameters
    ----------

    x: {array-like} of shape (n_samples, )
        Training vector, where `n_samples` is the number of samples

    cut: {array-like} of shape (n_samples, )
        Cutpoints of ReHU, where `n_samples` is the number of samples

    Returns
    -------
    array of shape (n_samples, )
        The result of the ReHU function.

    """
    n_samples = x.shape[0]
    cut = cut * np.ones_like(x)

    u = np.maximum(x, 0)
    return huber(cut, u)
