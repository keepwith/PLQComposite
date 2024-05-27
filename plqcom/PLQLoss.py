""" PLQLoss: Piecewise Linear Quadratic Loss function, with Decomposition to ReLU-ReHU Composition Loss functions """

# Author: Tingxian Gao <txgao@link.cuhk.edu.hk>
#         Ben Dai <bendai@cuhk.edu.hk>

# License: MIT License

import numpy as np
import itertools


class PLQLoss(object):
    """
    PLQLoss is a class represents a continuous convex piecewise quandratic loss function, which adopts three types of
    input forms: 'plq', 'max' and 'points'.

    Parameters
    ----------

    quad_coef : {dict-like} of {'a': [], 'b': [], 'c': []}
        The quandratic coefficients in pieces of the PLQLoss.
        The i-th piece Q is: a[i]* x**2 + b[i] * x + c[i]

    form : str, optional, default: 'plq'
        The form of the input PLQ function.

        'plq' for the PLQ form
            In this form, cutpoints must be given explicitly.

        'max' for the max form
            The max form is a special form of the PLQ function, which is the pointwise maximum of several linear or
            quadratic functions.
            The cutpoints are not necessary in this form, since they will be automatically calculated.

        'points' for the piecewise linear form based on given points
            The piecewise linear form is a special form of the PLQ function, which is the piecewise linear function.
            The function will connect the given points to form a piecewise linear loss.
            For the first piece and the last piece related to infinity, they will be the same as their adjacent piece.

    cutpoints : {array-like} of float, optional, default: None
        cutpoints of the PLQLoss, except -np.inf and np.inf

        if the form is 'max' or 'points', the cutpoints is not necessary

        if the form is 'plq', the cutpoints is necessary

    points : {array-like} of (x,y) pairs [(x1, y1), (x2, y2), ... (xn, yn)]
            or {dict-like} of {'x': [x1, x2, ..., xn], 'y': [y1, y2, ... yn]}
            or {2d-array-like} of [[x1, x2, ..., xn], [y1, y2, ... yn]]
            optional, default: None

        Points coordinates of the piecewise linear form of the PLQLoss. The PLQLoss will be constructed by straight
        lines between each two adjcent points according to their x coordinates. Two points with the same x coordinates
        will be rejected.

        if the form is 'points', the points is necessary

        if the form is 'max' or 'plq', the points is not necessary

    Examples
    --------
    >>> import numpy as np
    >>> from plqcom import PLQLoss
    >>> cutpoints = [0., 1.]
    >>> quad_coef = {'a': np.array([0., .5, 0.]), 'b': np.array([-1, 0., 1]), 'c': np.array([0., 0., -.5])}
    >>> random_loss = PLQLoss(quad_coef, cutpoints=cutpoints)
    >>> x = np.arange(-2,2,.05)
    >>> random_loss(x)
    """

    def __init__(self, quad_coef=None, form="plq", cutpoints=np.empty(shape=(0,)), points=np.empty(shape=(0, 2))):
        # check the quad_coef data type, if not np.array, convert it to np.array
        if quad_coef not in [None, {}]:
            if not isinstance(quad_coef['a'], np.ndarray):
                quad_coef['a'] = np.array(quad_coef['a'])
            if not isinstance(quad_coef['b'], np.ndarray):
                quad_coef['b'] = np.array(quad_coef['b'])
            if not isinstance(quad_coef['c'], np.ndarray):
                quad_coef['c'] = np.array(quad_coef['c'])
            if not isinstance(cutpoints, np.ndarray):
                cutpoints = np.array(cutpoints)

            # check the quad_coef length
            if len(quad_coef['a']) != len(quad_coef['b']) or len(quad_coef['a']) != len(quad_coef['c']):
                print("The size of `quad_coef` is not matched!")
                exit()

        # check the input form
        if form not in ['plq', 'max', 'points']:
            print("The input form of PLQ function is not supported!")
            exit()

        # max form input
        if form == "max":
            self.cutpoints = []
            self.quad_coef, self.cutpoints, self.n_pieces = max_to_plq(quad_coef)

        # PLQ form input
        elif form == 'plq':
            # check whether the cutpoints are given
            if cutpoints.size == 0 and len(quad_coef['a']) != 1:
                print("The `cutpoints` is not given!")
                exit()
            elif len(cutpoints) != (len(quad_coef['a']) - 1):
                print("The size of cutpoints and quad_coef is not matched!")
                exit()

            self.cutpoints = cutpoints
            self.quad_coef = quad_coef
            self.n_pieces = len(self.quad_coef['a'])

        # piecewise linear form input
        elif form == 'points':
            # check the length of input points
            if len(points) < 2:
                print("Input points are not given or not enough!")
                exit()
            else:
                self.cutpoints, self.quad_coef, self.n_pieces = points_to_plq(points)

        self.quad_coef, self.cutpoints, self.n_pieces = merge_successive_intervals(self.quad_coef, self.cutpoints)

        # initialize the minimum value and minimum knot
        self.cutpoints = np.concatenate(([-np.inf], self.cutpoints, [np.inf]))
        self.min_val = np.inf
        self.min_knot = np.inf

    def __call__(self, x):
        """
        Evaluation of PLQLoss function.

        Parameters
        ----------
        x : {array-like} of shape {n_samples}
        Training vector, where `n_samples` is the number of samples.

        Returns
        -------
        y : {array-like} of shape {n_samples}
            The values of the PLQLoss function on each x
            y[j] = quad_coef['a'][i]*x[j]**2 + quad_coef['b'][i]*x[j] + quad_coef['c'][i],
            if cutpoints[i] < x[j] < cutpoints[i+1]
        """

        x = np.array(x)
        # check the size of coefficients
        assert len(self.quad_coef['a']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."
        assert len(self.quad_coef['b']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."
        assert len(self.quad_coef['c']) == self.n_pieces, "`cutpoints` and `quad_coef` are mismatched."

        y = np.zeros_like(x)
        for i in range(self.n_pieces):
            cond_tmp = (x > self.cutpoints[i]) & (x <= self.cutpoints[i + 1])
            y[cond_tmp] = (self.quad_coef['a'][i] * x[cond_tmp] ** 2 + self.quad_coef['b'][i] * x[cond_tmp] +
                           self.quad_coef['c'][i])

        # add back the minimum value
        if self.min_val != np.inf:
            return y + self.min_val
        else:
            return y


def max_to_plq(quad_coef):
    # convert the max form to the PLQ form
    diff_a = np.diff(np.array(list(itertools.combinations(quad_coef['a'], 2))))
    diff_b = np.diff(np.array(list(itertools.combinations(quad_coef['b'], 2))))
    diff_c = np.diff(np.array(list(itertools.combinations(quad_coef['c'], 2))))
    index_1 = np.logical_and(diff_a == 0, diff_b != 0)
    sol1 = -diff_c[index_1] / diff_b[index_1]
    index_2 = np.logical_and(diff_a != 0, diff_b * diff_b - 4 * diff_a * diff_c >= 0)
    sol2 = (-diff_b[index_2] + np.sqrt(
        diff_b[index_2] * diff_b[index_2] - 4 * diff_a[index_2] * diff_c[index_2])) / (2 * diff_a[index_2])
    sol3 = (-diff_b[index_2] - np.sqrt(
        diff_b[index_2] * diff_b[index_2] - 4 * diff_a[index_2] * diff_c[index_2])) / (2 * diff_a[index_2])

    # remove duplicate solutions
    cutpoints = np.sort(np.array(list(set(np.concatenate((sol1, sol2, sol3)).tolist())), dtype=float))

    if len(cutpoints) == 0:
        ind_tmp = np.argmax(quad_coef['c'])  # just compare the function value at x = 0
        new_quad_coef = {'a': np.array([quad_coef['a'][ind_tmp]]), 'b': np.array([quad_coef['b'][ind_tmp]]),
                         'c': np.array([quad_coef['c'][ind_tmp]])}
        new_cutpoints = np.array([])
        new_n_pieces = 1
    else:
        evals = (cutpoints[:-1] + cutpoints[1:]) / 2
        evals = np.concatenate(([-1 + cutpoints[0]], evals, [1 + cutpoints[-1]]))
        new_quad_coef = {'a': np.array([]), 'b': np.array([]), 'c': np.array([])}
        for i in range(len(evals)):
            ind_tmp = np.argmax(quad_coef['a'] * evals[i] ** 2 + quad_coef['b'] * evals[i] + quad_coef['c'])
            new_quad_coef['a'] = np.append(new_quad_coef['a'], quad_coef['a'][ind_tmp])
            new_quad_coef['b'] = np.append(new_quad_coef['b'], quad_coef['b'][ind_tmp])
            new_quad_coef['c'] = np.append(new_quad_coef['c'], quad_coef['c'][ind_tmp])
        new_cutpoints = cutpoints
        new_n_pieces = len(new_quad_coef['a'])

    return new_quad_coef, new_cutpoints, new_n_pieces


def points_to_plq(points):
    # convert the points form to the PLQ form
    x, y = [], []
    if type(points) == dict:
        x, y = np.array(points['x']), np.array(points['y'])
    elif type(points) == np.ndarray:
        if points.shape[0] == 2:
            x, y = points[0, :], points[1, :]
        elif points.shape[1] == 2:
            x, y = points[:, 0], points[:, 1]
    else:
        print("The input points form is not supported!")
        exit()

    # check the x coordinates
    if len(x) != len(set(x)):
        print("Duplicated x! Input ")
        exit()

    # sort and calculate the quad_coef
    x, y = zip(*sorted(zip(x, y)))
    b = np.diff(y) / np.diff(x)
    c = y[1:] - b * x[1:]
    b = np.concatenate(([b[0]], b, [b[-1]]))
    c = np.concatenate(([c[0]], c, [c[-1]]))
    cutpoints = x
    quad_coef = {'a': np.zeros_like(b), 'b': b, 'c': c}
    n_pieces = len(b)
    return cutpoints, quad_coef, n_pieces


def merge_successive_intervals(quad_coef, cutpoints):
    # merge the successive intervals with the same coefficients
    i = 0
    while i < len(quad_coef['a']) - 1:
        if (quad_coef['a'][i] == quad_coef['a'][i + 1] and
                quad_coef['b'][i] == quad_coef['b'][i + 1] and
                quad_coef['c'][i] == quad_coef['c'][i + 1]):
            quad_coef['a'] = np.delete(quad_coef['a'], i + 1)
            quad_coef['b'] = np.delete(quad_coef['b'], i + 1)
            quad_coef['c'] = np.delete(quad_coef['c'], i + 1)
            cutpoints = np.delete(cutpoints, i)
        else:
            i += 1
    n_pieces = len(quad_coef['a'])
    return quad_coef, cutpoints, n_pieces
