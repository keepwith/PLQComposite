""" PLQProperty: Several functions to check the properties of a PLQ function. """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>

# License: MIT License

import numpy as np
from plqcom.PLQLoss import PLQLoss
from rehline import ReHLoss


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
    >>> from plqcom.PLQLoss import PLQLoss
    >>> from plqcom.PLQProperty import is_continuous
    >>> plqloss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]),
    >>>                                                     'c': np.array([1, 1])})
    >>> is_continuous(plqloss)
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
    >>> from plqcom.PLQLoss import PLQLoss
    >>> from plqcom.PLQProperty import is_convex
    >>> plqloss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]),
    >>>                                                     'c': np.array([1, 1])})
    >>> is_convex(plqloss)
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
    >>> from plqcom.PLQLoss import PLQLoss
    >>> from plqcom.PLQProperty import check_cutoff
    >>> plqloss = PLQLoss(cutpoints=np.array([]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> check_cutoff(plqloss)
    >>> plq_loss.cutpoints
    >>> [0.]
    """
    cutpoints = plq_loss.cutpoints.copy()
    quad_coef = plq_loss.quad_coef.copy()

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
    >>> from plqcom.PLQLoss import PLQLoss
    >>> from plqcom.PLQProperty import find_min
    >>> plqloss = PLQLoss(cutpoints=np.array([0]),quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
    >>> find_min(plqloss)
    >>> plq_loss.min_val
    >>> 1
    """
    # find the minimum value and knot
    out_cut = plq_loss.quad_coef['a'][:-1] * plq_loss.cutpoints[1:-1] ** 2 + (
            plq_loss.quad_coef['b'][:-1] * plq_loss.cutpoints[1:-1] + plq_loss.quad_coef['c'][:-1])
    # out_cut = plq_loss(plq_loss.cutpoints[1:-1])
    plq_loss.min_val = np.min(out_cut)
    plq_loss.min_knot = np.argmin(out_cut) + 1
    # remove self.min_val from the PLQ function
    plq_loss.quad_coef['c'] = plq_loss.quad_coef['c'] + (-plq_loss.min_val)


def plq_to_rehloss(plq_loss):
    """convert the PLQ function to a ReHLoss function

    Returns
    -------
        an object of ReHLoss

    """

    # check the continuity and convexity of the PLQ function
    if not is_continuous(plq_loss):
        print("The PLQ function is not continuous!")
        exit()

    if not is_convex(plq_loss):
        print("The PLQ function is not convex!")
        exit()

    # check the cutoff of each piece
    check_cutoff(plq_loss)

    # find the minimum value and knot
    find_min(plq_loss)

    quad_coef = plq_loss.quad_coef.copy()
    cutpoints = plq_loss.cutpoints.copy()

    # remove a ReLU/ReHU function from this point; i-th point -> i-th or (i+1)-th interval
    ind_tmp = plq_loss.min_knot

    # Right
    cutpoints_r = cutpoints[ind_tmp:]
    quad_coef_r = {'a': quad_coef['a'][ind_tmp:], 'b': quad_coef['b'][ind_tmp:], 'c': quad_coef['c'][ind_tmp:]}
    # +relu
    relu_coef_r = 2 * np.diff(quad_coef_r['a'], prepend=0) * cutpoints_r[:-1] + np.diff(
        quad_coef_r['b'], prepend=0)
    relu_intercept_r = -relu_coef_r * cutpoints_r[:-1]
    #  remove all zero coefficients terms
    relu_intercept_r = relu_intercept_r[relu_coef_r != 0]
    relu_coef_r = relu_coef_r[relu_coef_r != 0]

    # +rehu
    rehu_coef_r = np.sqrt(2 * quad_coef_r['a'])
    rehu_intercept_r = -np.sqrt(2 * quad_coef_r['a']) * cutpoints_r[:-1]
    cut_diff_r = np.diff(cutpoints_r)[quad_coef_r['a'] != 0]
    quad_coef_r['a'] = quad_coef_r['a'][quad_coef_r['a'] != 0]
    rehu_cut_r = np.sqrt(2 * quad_coef_r['a'] * cut_diff_r)

    rehu_intercept_r = rehu_intercept_r[rehu_coef_r != 0]
    rehu_coef_r = rehu_coef_r[rehu_coef_r != 0]

    # Left
    cutpoints_l = np.flip(cutpoints[:ind_tmp + 1])
    quad_coef_l = {'a': np.flip(quad_coef['a'][:ind_tmp]), 'b': np.flip(quad_coef['b'][:ind_tmp]),
                   'c': np.flip(quad_coef['c'][:ind_tmp])}
    # +relu
    relu_coef_l = 2 * np.diff(quad_coef_l['a'], prepend=0) * cutpoints_l[:-1] + np.diff(
        quad_coef_l['b'], prepend=0)
    relu_intercept_l = -relu_coef_l * cutpoints_l[:-1]

    relu_intercept_l = relu_intercept_l[relu_coef_l != 0]
    relu_coef_l = relu_coef_l[relu_coef_l != 0]

    # +rehu
    rehu_coef_l = -np.sqrt(2 * quad_coef_l['a'])
    rehu_intercept_l = np.sqrt(2 * quad_coef_l['a']) * cutpoints_l[:-1]
    cut_diff_l = (cutpoints_l[:-1] - cutpoints_l[1:])[quad_coef_l['a'] != 0]
    quad_coef_l['a'] = quad_coef_l['a'][quad_coef_l['a'] != 0]
    rehu_cut_l = np.sqrt(2 * quad_coef_l['a'] * cut_diff_l)

    rehu_intercept_l = rehu_intercept_l[rehu_coef_l != 0]
    rehu_coef_l = rehu_coef_l[rehu_coef_l != 0]

    return ReHLoss(relu_coef=np.concatenate((relu_coef_l, relu_coef_r)).reshape((-1, 1)),
                   relu_intercept=np.concatenate((relu_intercept_l, relu_intercept_r)).reshape((-1, 1)),
                   rehu_coef=np.concatenate((rehu_coef_l, rehu_coef_r)).reshape((-1, 1)),
                   rehu_intercept=np.concatenate((rehu_intercept_l, rehu_intercept_r)).reshape((-1, 1)),
                   rehu_cut=np.concatenate((rehu_cut_l, rehu_cut_r)).reshape((-1, 1)))
