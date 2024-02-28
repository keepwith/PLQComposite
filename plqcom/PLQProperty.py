""" PLQProperty: Several functions to check the properties of a PLQ function. """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>

# License: MIT License

import numpy as np


def is_continuous(plq_loss):
    """
        Check whether a PLQ loss function is continuous

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Returns
    -------
    bool
        Whether the PLQ function is continuous, True for continuous, False for not continuous

    """
    cutpoints = plq_loss.cutpoints[1:-1].copy()
    quad_coef = plq_loss.quad_coef.copy()
    n_pieces = plq_loss.n_pieces

    # check the continuity at cut points from left to right
    for i in range(n_pieces - 1):
        if (quad_coef['a'][i] * (cutpoints[i] ** 2) + quad_coef['b'][i] *
            cutpoints[i] + quad_coef['c'][i]) - (quad_coef['a'][i + 1] *
                                                 (cutpoints[i] ** 2) + quad_coef['b'][i + 1] * cutpoints[i] +
                                                 quad_coef['c'][i + 1]) > 1e-6:
            return False

    return True


def is_convex(plq_loss):
    """
        Check whether a PLQ loss function is convex

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Returns
    -------
    bool
        Whether the PLQ function is convex, True for convex, False for not convex

    """
    # check the second order derivatives
    if np.min(plq_loss.quad_coef['a']) < 0:
        return False

    # compare the first order derivatives at cut points
    for i in range(plq_loss.n_pieces - 1):
        if (2 * plq_loss.quad_coef['a'][i] * plq_loss.cutpoints[i + 1] + plq_loss.quad_coef['b'][i] >
                2 * plq_loss.quad_coef['a'][i + 1] * plq_loss.cutpoints[i + 1] + plq_loss.quad_coef['b'][i + 1]):
            return False

    return True


def check_cutoff(plq_loss):
    """
        Check whether there exists a cutoff between the knots, if so, add the cutoff to the knot list and update
        the coefficients

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    """
    cutpoints = plq_loss.cutpoints.copy()
    quad_coef = plq_loss.quad_coef.copy()
    n_pieces = plq_loss.n_pieces

    # check the cutoff of each piece
    i = 0

    while i < len(quad_coef['a']):
        if quad_coef['a'][i] != 0:  # only will happen when the quadratic term is not zero
            middlepoint = -quad_coef['b'][i] / (2 * quad_coef['a'][i])
            if cutpoints[i] < middlepoint < cutpoints[i + 1]:  # if the cutoff is between the knots
                # add the cutoff to the knot list and update the coefficients
                cutpoints = np.insert(cutpoints, i + 1, middlepoint)
                quad_coef['a'] = np.insert(quad_coef['a'], i + 1, quad_coef['a'][i])
                quad_coef['b'] = np.insert(quad_coef['b'], i + 1, quad_coef['b'][i])
                quad_coef['c'] = np.insert(quad_coef['c'], i + 1, quad_coef['c'][i])
        i += 1

    plq_loss.cutpoints = cutpoints
    plq_loss.quad_coef = quad_coef
    plq_loss.n_pieces = len(quad_coef['a'])


def find_min(plq_loss):
    """
        Find the minimum knots and value of the PLQ function, if the minimum value is greater than zero
        record the minimum value and knot, remove the minimum value from the PLQ function and update the coefficients

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    """
    # find the minimum value and knot
    out_cut = plq_loss(plq_loss.cutpoints[1:-1])
    plq_loss.min_val = np.min(out_cut)
    plq_loss.min_knot = np.argmin(out_cut) + 1
    # remove self.min_val from the PLQ function
    plq_loss.quad_coef['c'] = plq_loss.quad_coef['c'] + (-plq_loss.min_val)
