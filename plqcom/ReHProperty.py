""" ReHProperty: Several functions to check or perform the properties of ReHLoss. """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>

# License: MIT License

import numpy as np
from rehline._loss import ReHLoss


def affine_transformation(rehloss=ReHLoss, c=1, p=1, q=0):
    """
        Since composite ReLU-ReHU function is closure under affine transformation,
        this function perform affine transformation on the PLQ object

    Parameters
    ----------
    rehloss : ReHLoss
         A ReHLoss object
    c: scale parameter on loss function and require c > 0
    p: scale parameter on x
    q: shift parameter on x

    Returns
    -------
    ReHLoss
        A ReHLoss object after affine transformation

    """
    loss = rehloss.copy()

    if not loss.U:
        print("The ReH presentation is empty!")
    elif c <= 0:
        raise Exception("c must greater than 0!")
    else:
        loss.U = c * p * loss.U
        loss.V = c * loss.U * q + c * loss.V
        loss.Tau = np.sqrt(c) * loss.Tau
        loss.S = np.sqrt(c) * p * loss.S
        loss.T = np.sqrt(c) * (loss.S * q + loss.T)

    return loss


def make_rehloss():
    """
        Make a ReHLoss object after transform a PLQ Loss to ReLU-ReHU composition

    Returns
    -------
    ReHLoss
        A ReHLoss object

    """
    return ReHLoss()