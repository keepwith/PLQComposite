""" PLQProperty: Several functions to check the properties of a PLQ function. """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>

# License: MIT License

import numpy as np


def is_continuous(plq_loss):
    """Check whether a PLQ loss function is continuous

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Returns
    -------
    bool
        Whether the PLQ function is continuous, True for continuous, False for not continuous

    Examples
    --------
    >>> from plqcom import PLQLoss
    >>> plq_loss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> PLQProperty.is_continuous(plq_loss)
    >>> True
    """

    cutpoints = plq_loss.cutpoints[1:-1].copy()
    quad_coef = plq_loss.quad_coef.copy()

    # only one piece
    if plq_loss.n_pieces == 1:
        return True

    diff_max = np.max(
        np.diff(quad_coef['a']) * (cutpoints ** 2) + np.diff(quad_coef['b']) * cutpoints + np.diff(quad_coef['c']))
    diff_min = np.min(
        np.diff(quad_coef['a']) * (cutpoints ** 2) + np.diff(quad_coef['b']) * cutpoints + np.diff(quad_coef['c']))

    if diff_max > 1e-6 or diff_min < -1e-6:
        return False
    return True


def is_convex(plq_loss):
    """Check whether a PLQ loss function is convex

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Returns
    -------
    bool
        Whether the PLQ function is convex, True for convex, False for not convex

    Examples
    --------
    >>> from plqcom import PLQLoss
    >>> plq_loss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> PLQProperty.is_convex(plq_loss)
    >>> True
    """
    # check the second order derivatives
    if np.min(plq_loss.quad_coef['a']) < 0:
        return False

    # only one piece
    if plq_loss.n_pieces == 1:
        return True

    # check the minimum value of $2(a_i - a_{i-1}) * x_{i-1} + (b_i - b_{i-1})$
    if np.min(2 * np.diff(plq_loss.quad_coef['a']) * plq_loss.cutpoints[1:-1] + np.diff(plq_loss.quad_coef['b'])) < 0:
        return False

    return True


def check_cutoff(plq_loss):
    """Check whether there exists a cutoff between the knots, if so, add the cutoff to the knot list and update
    the coefficients

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Examples
    --------
    >>> from plqcom import PLQLoss
    >>> plq_loss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> PLQProperty.check_cutoff(plq_loss)
    >>> plq_loss.cutpoints
    >>> [0.]
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
    """Find the minimum knots and value of the PLQ function, if the minimum value is greater than zero
    record the minimum value and knot, remove the minimum value from the PLQ function and update the coefficients

    Parameters
    ----------
    plq_loss : PLQLoss
         A PLQLoss object

    Examples
    --------
    >>> from plqcom import PLQLoss
    >>> plq_loss = PLQLoss(cutpoints=np.array([0]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> PLQProperty.find_min(plq_loss)
    >>> plq_loss.min_val
    >>> 1
    """
    # find the minimum value and knot
    out_cut = plq_loss(plq_loss.cutpoints[1:-1])
    plq_loss.min_val = np.min(out_cut)
    plq_loss.min_knot = np.argmin(out_cut) + 1
    # remove self.min_val from the PLQ function
    plq_loss.quad_coef['c'] = plq_loss.quad_coef['c'] + (-plq_loss.min_val)
