""" PLQLoss: Piecewise Linear Quadratic Loss function, with Decomposition to ReLU-ReHU Composition Loss functions """

# Author: Tingxian Gao <txgao@cuhk.edu.hk>
#         Ben Dai <bendai@cuhk.edu.hk>

# License: MIT License

import numpy as np
from plqcom import PLQProperty
from rehline._loss import ReHLoss
from sympy import symbols, solve, Eq


class PLQLoss(object):
    """
    PLQLoss: continuous convex piecewise quandratic function (with a function converting to ReHLoss).

    Parameters
    ----------

    quad_coef : {dict-like} of {'a': [], 'b': [], 'c': []}
        The quandratic coefficients in pieces of the PQLoss
        The i-th piece Q is: a[i]* x**2 + b[i] * x + c[i]

    form : str, optional, default: 'plq'
        The form of the input PLQ function
        'plq' for the PLQ form,
            In this form, cutpoints must be given explicitly
        'minimax' for the minimax form
            The minimax form is a special form of the PLQ function, which is the maximum of several quadratic functions.
            The cutpoints are not necessary in this form. The cutpoints will be automatically calculated.

    cutpoints : {array-like} of float, optional, default: None
        cutpoints of the PQLoss, except -np.inf and np.inf
        if the form is 'minimax', the cutpoints is not necessary
        if the form is 'plq', the cutpoints is necessary

    Example
    -------
    >>> import numpy as np
    >>> cutpoints = [0., 1.]
    >>> quad_coef = {'a': np.array([0., .5, 0.]), 'b': np.array([-1, 0., 1]), 'c': np.array([0., 0., -.5])}
    >>> test_loss = PQLoss(quad_coef, cutpoints=cutpoints)
    >>> x = np.arange(-2,2,.05)
    >>> test_loss(x)
    """

    def __init__(self, quad_coef, form="plq", cutpoints=np.empty(shape=(0,))):
        # check the input data type, if not np.array, convert it to np.array
        # needed to be fixed further
        if not isinstance(quad_coef['a'], np.ndarray):
            quad_coef['a'] = np.array(quad_coef['a'])
        if not isinstance(quad_coef['b'], np.ndarray):
            quad_coef['b'] = np.array(quad_coef['b'])
        if not isinstance(quad_coef['c'], np.ndarray):
            quad_coef['c'] = np.array(quad_coef['c'])
        if not isinstance(cutpoints, np.ndarray):
            cutpoints = np.array(cutpoints)

        # check the quad_coef
        if len(quad_coef['a']) != len(quad_coef['b']) or len(quad_coef['a']) != len(quad_coef['c']):
            print("The size of `quad_coef` is not matched!")
            exit()

        # check the input form
        if form not in ['plq', 'minimax']:
            print("The input form of PLQ function is not supported!")
            exit()

        # minimax form input
        if form == "minimax":
            self.quad_coef, self.cutpoints, self.n_pieces = self.minimax2plq(quad_coef)
            self.cutpoints = np.concatenate(([-np.inf], self.cutpoints, [np.inf]))
            self.min_val = np.inf
            self.min_knot = np.inf

        # PLQ form input
        elif form == 'plq':
            # check whether the cutpoints are given
            if cutpoints.size == 0 and len(quad_coef['a']) != 1:
                print("The `cutpoints` is not given!")
                exit()
            elif len(cutpoints) != (len(quad_coef['a']) - 1):
                print("The size of cutpoints and quad_coef is not matched!")
                exit()
            else:
                self.cutpoints = np.concatenate(([-np.inf], cutpoints, [np.inf]))
            self.quad_coef = quad_coef
            self.n_pieces = len(self.quad_coef['a'])
            self.min_val = np.inf
            self.min_knot = np.inf

    def __call__(self, x):
        """ Evaluation of PQLoss

        out = quad_coef['a'][i]*x**2 + quad_coef['b'][i]*x + quad_coef['c'][i], if cutpoints[i] < x < cutpoints[i+1]
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

    def minimax2plq(self, quad_coef):
        """
            convert the minimax form to the PLQ form
        """
        solutions = np.array([])
        n_pieces = len(quad_coef['a'])
        x = symbols('x', real=True)
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                solutions = np.append(solutions,
                                      solve(Eq(quad_coef['a'][i] * x ** 2 + quad_coef['b'][i] * x + quad_coef['c'][i],
                                               quad_coef['a'][j] * x ** 2 + quad_coef['b'][j] * x + quad_coef['c'][j]),
                                            x))

        solutions = list(set(solutions.tolist()))  # remove the duplicate solutions
        cutpoints = np.sort(np.array(solutions, dtype=float))

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

            # merge the successive intervals with the same coefficients
            i = 0
            while i < len(new_quad_coef['a']) - 1:
                if (new_quad_coef['a'][i] == new_quad_coef['a'][i + 1] and
                        new_quad_coef['b'][i] == new_quad_coef['b'][i + 1] and
                        new_quad_coef['c'][i] == new_quad_coef['c'][i + 1]):
                    new_quad_coef['a'] = np.delete(new_quad_coef['a'], i + 1)
                    new_quad_coef['b'] = np.delete(new_quad_coef['b'], i + 1)
                    new_quad_coef['c'] = np.delete(new_quad_coef['c'], i + 1)
                    cutpoints = np.delete(cutpoints, i)
                else:
                    i += 1

            new_cutpoints = cutpoints
            new_n_pieces = len(new_quad_coef['a'])

        return new_quad_coef, new_cutpoints, new_n_pieces

    def _2ReHLoss(self):
        """
            convert the PLQ function to a ReHLoss function

        :return:
            an object of ReHLoss
        """

        # check the continuity and convexity of the PLQ function
        if not PLQProperty.is_continuous(self):
            print("The PLQ function is not continuous!")
            exit()

        if not PLQProperty.is_convex(self):
            print("The PLQ function is not convex!")
            exit()

        # check the cutoff of each piece
        PLQProperty.check_cutoff(self)

        # find the minimum value and knot
        PLQProperty.find_min(self)

        relu_coef, relu_intercept = [], []
        rehu_coef, rehu_intercept, rehu_cut = [], [], []
        quad_coef = self.quad_coef.copy()
        cutpoints = self.cutpoints.copy()

        # remove a ReLU/ReHU function from this point; i-th point -> i-th or (i+1)-th interval
        ind_tmp = self.min_knot

        # Right
        # first interval on the right
        # + relu
        temp = 2 * quad_coef['a'][ind_tmp] * cutpoints[ind_tmp] + quad_coef['b'][ind_tmp]
        if temp != 0:
            relu_coef.append(temp)
            relu_intercept.append(-temp * cutpoints[ind_tmp])

        if quad_coef['a'][ind_tmp] != 0:
            # +rehu
            rehu_coef.append(np.sqrt(2 * quad_coef['a'][ind_tmp]))
            rehu_intercept.append(-np.sqrt(2 * quad_coef['a'][ind_tmp]) * cutpoints[ind_tmp])
            rehu_cut.append(np.sqrt(2 * quad_coef['a'][ind_tmp]) * (cutpoints[ind_tmp + 1] - cutpoints[ind_tmp]))

        # other intervals on the right
        for i in range(ind_tmp + 1, len(cutpoints) - 1):
            # +relu
            temp = 2 * (quad_coef['a'][i] - quad_coef['a'][i - 1]) * cutpoints[i] + (
                    quad_coef['b'][i] - quad_coef['b'][i - 1])
            if temp != 0:
                relu_coef.append(temp)
                relu_intercept.append(-temp * cutpoints[i])

            if quad_coef['a'][i] != 0:
                # +rehu
                rehu_coef.append(np.sqrt(2 * quad_coef['a'][i]))
                rehu_intercept.append(-np.sqrt(2 * quad_coef['a'][i]) * cutpoints[i])
                rehu_cut.append(np.sqrt(2 * quad_coef['a'][i]) * (cutpoints[i + 1] - cutpoints[i]))

        # Left
        # first interval on the left
        # + relu
        temp = 2 * quad_coef['a'][ind_tmp - 1] * cutpoints[ind_tmp] + quad_coef['b'][ind_tmp - 1]
        if temp != 0:
            relu_coef.append(temp)
            relu_intercept.append(-temp * cutpoints[ind_tmp])

        if quad_coef['a'][ind_tmp - 1] != 0:
            # +rehu
            rehu_coef.append(-np.sqrt(2 * quad_coef['a'][ind_tmp - 1]))
            rehu_intercept.append(np.sqrt(2 * quad_coef['a'][ind_tmp - 1]) * cutpoints[ind_tmp])
            rehu_cut.append(np.sqrt(2 * quad_coef['a'][ind_tmp - 1]) * (cutpoints[ind_tmp] - cutpoints[ind_tmp - 1]))

        # other intervals on the left
        for i in range(0, ind_tmp - 1):
            # +relu
            temp = 2 * (quad_coef['a'][i] - quad_coef['a'][i + 1]) * cutpoints[i + 1] + (
                    quad_coef['b'][i] - quad_coef['b'][i + 1])
            if temp != 0:
                relu_coef.append(temp)
                relu_intercept.append(-temp * cutpoints[i + 1])

            if quad_coef['a'][i] != 0:
                # +rehu
                rehu_coef.append(-np.sqrt(2 * quad_coef['a'][i]))
                rehu_intercept.append(np.sqrt(2 * quad_coef['a'][i]) * cutpoints[i + 1])
                rehu_cut.append(np.sqrt(2 * quad_coef['a'][i]) * (cutpoints[i + 1] - cutpoints[i]))

        return ReHLoss(relu_coef=np.array(relu_coef).reshape((-1, 1)),
                       relu_intercept=np.array(relu_intercept).reshape((-1, 1)),
                       rehu_coef=np.array(rehu_coef).reshape((-1, 1)),
                       rehu_intercept=np.array(rehu_intercept).reshape((-1, 1)),
                       rehu_cut=np.array(rehu_cut).reshape((-1, 1)))
