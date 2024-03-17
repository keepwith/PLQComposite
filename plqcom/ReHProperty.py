""" ReHProperty: Several functions to check or perform the properties of ReHLoss. """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>

# License: MIT License

import numpy as np
from rehline._loss import ReHLoss


def affine_transformation(rehloss: ReHLoss, n=1, c=1, p=1, q=0):
    """Since composite ReLU-ReHU function is closure under affine transformation,
    this function perform affine transformation on the PLQ object

    Parameters
    ----------
    rehloss : ReHLoss
         A ReHLoss object
    c: a number or {array_like} of shape (n_samples,), default=1
        scale parameter on loss function and require c > 0
    p: a number or {array_like} of shape (n_samples,),default=1
        scale parameter on z
    q: a number or {array_like} of shape (n_samples,),default=0
        shift parameter on z
    n: int, default=1
        number of samples

    Returns
    -------
    ReHLoss
        A ReHLoss object after affine transformation

    Examples
    --------
    >>> from plqcom.PLQLoss import PLQLoss
    >>> from plqcom.ReHProperty import affine_transformation
    >>> from plqcom.PLQProperty import plq_to_rehloss
    >>> import numpy as np
    >>> from rehline import ReHLine
    >>> n, d, C = 1000, 3, 0.5
    >>> np.random.seed(1024)
    >>> X = np.random.randn(1000, 3)
    >>> beta0 = np.random.randn(3)
    >>> y = np.sign(X.dot(beta0) + np.random.randn(n))
    >>> plqloss = PLQLoss(quad_coef={'a': np.array([0., 0.]), 'b': np.array([0., 1.]), 'c': np.array([0., 0.])}, cutpoints=np.array([0]))
    >>> rehloss = plq_to_rehloss(plqloss)
    >>> rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=-y, q=1)
    """

    loss = ReHLoss(relu_coef=rehloss.relu_coef, relu_intercept=rehloss.relu_intercept,
                   rehu_coef=rehloss.rehu_coef, rehu_intercept=rehloss.rehu_intercept, rehu_cut=rehloss.rehu_cut)

    # check if the ReHLoss presentation is empty
    if loss.relu_coef.size == 0 and loss.rehu_coef.size == 0:
        raise Exception("The ReHLoss presentation is empty!")

    # check if the ReH presentation is a single loss. if true, then broadcast it
    if loss.n < n:
        loss.relu_coef = np.tile(loss.relu_coef, (1, n))
        loss.relu_intercept = np.tile(loss.relu_intercept, (1, n))
        loss.rehu_cut = np.tile(loss.rehu_cut, (1, n))
        loss.rehu_coef = np.tile(loss.rehu_coef, (1, n))
        loss.rehu_intercept = np.tile(loss.rehu_intercept, (1, n))

    # perform affine transformation
    if c <= 0:
        raise Exception("c must greater than 0!")
    else:
        # each c p q can be either a number or an array
        relu_coef = c * p * loss.relu_coef
        relu_intercept = c * loss.relu_coef * q + c * loss.relu_intercept
        rehu_cut = np.sqrt(c) * loss.rehu_cut
        rehu_coef = np.sqrt(c) * p * loss.rehu_coef
        rehu_intercept = np.sqrt(c) * (loss.rehu_coef * q + loss.rehu_intercept)

    return ReHLoss(relu_coef=relu_coef, relu_intercept=relu_intercept, rehu_coef=rehu_coef,
                   rehu_intercept=rehu_intercept, rehu_cut=rehu_cut)
