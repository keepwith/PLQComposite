import numpy as np
from plqutils import PLQProperty
from plqcom.ReHLoss import ReHLoss
from sympy import symbols, solve, Eq


class PLQLoss(object):
    """
        Piecewise Linear Quadratic Loss function
    """

    def __init__(self, quad_coef, type="plq", **paras):
        # check the quad_coef
        if len(quad_coef['a']) != len(quad_coef['b']) or len(quad_coef['a']) != len(quad_coef['c']):
            print("The size of `quad_coef` is not matched!")
            exit()

        # check the type
        if type not in ['plq', 'minimax']:
            print("The type of PLQ function is not supported!")
            exit()

        # minimax form input
        if type == "minimax":
            self.quad_coef, self.cutpoints, self.n_pieces = self.minimax2plq(quad_coef)
            self.min_val = np.inf
            self.min_knot = np.inf

        # PLQ form input
        elif type == 'plq':
            # check whether the cutpoints are given
            if 'cutpoints' not in paras.keys():
                print("The `cutpoints` is not given!")
                exit()
            else:
                self.cutpoints = np.concatenate(([-np.inf], paras['cutpoints'], [np.inf]))
            self.quad_coef = quad_coef
            self.n_pieces = len(self.cutpoints) - 1
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
        solutions = []
        n_pieces = len(quad_coef['a'])
        x = symbols('x', real=True)
        for i in range(n_pieces):
            for j in range(i + 1, n_pieces):
                solutions += solve(Eq(quad_coef['a'][i] * x ** 2 + quad_coef['b'][i] * x + quad_coef['c'][i],
                                      quad_coef['a'][j] * x ** 2 + quad_coef['b'][j] * x + quad_coef['c'][j]), x)

        solutions = list(set(solutions))  # remove the duplicate solutions
        cutpoints = np.sort(np.array(solutions))

        if len(cutpoints) == 0:
            ind_tmp = np.argmax(quad_coef['c'])  # just compare the function value at x=0
            new_quad_coef = {'a': np.array([quad_coef['a'][ind_tmp]]), 'b': np.array([quad_coef['b'][ind_tmp]]),
                             'c': np.array([quad_coef['c'][ind_tmp]])}
            new_cutpoints = np.array([])
            new_n_pieces = 1
        else:
            evals = (cutpoints[:-1] + cutpoints[1:]) / 2
            evals = np.concatenate(([-1 + cutpoints[0]], evals, 1 + cutpoints[-1]))
            new_quad_coef = {'a': np.array([]), 'b': np.array([]), 'c': np.array([])}
            for i in range(len(evals) - 1):
                ind_tmp = np.argmin(quad_coef['a'] * evals[i] ** 2 + quad_coef['b'] * evals[i] + quad_coef['c'])
                new_quad_coef['a'] = np.append(new_quad_coef['a'], quad_coef['a'][ind_tmp])
                new_quad_coef['b'] = np.append(new_quad_coef['b'], quad_coef['b'][ind_tmp])
                new_quad_coef['c'] = np.append(new_quad_coef['c'], quad_coef['c'][ind_tmp])

            # merge the successive intervals with the same coefficients
            i = 0
            while i < len(new_quad_coef['a']) - 1:
                if (new_quad_coef['a'][i] == new_quad_coef['a'][i + 1] and new_quad_coef['b'][i] == new_quad_coef['b'][
                        i + 1] and new_quad_coef['c'][i] == new_quad_coef['c'][i + 1]):
                    new_quad_coef['a'] = np.delete(new_quad_coef['a'], i + 1)
                    new_quad_coef['b'] = np.delete(new_quad_coef['b'], i + 1)
                    new_quad_coef['c'] = np.delete(new_quad_coef['c'], i + 1)
                    cutpoints = np.delete(cutpoints, i)

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
        relu_coef.append(temp)
        relu_intercept.append(-temp * cutpoints[ind_tmp])

        if quad_coef['a'][ind_tmp] != 0:
            # +rehu
            rehu_coef.append(np.sqrt(-2 * quad_coef['a'][ind_tmp - 1]))
            rehu_intercept.append(np.sqrt(2 * quad_coef['a'][ind_tmp - 1]) * cutpoints[ind_tmp])
            rehu_cut.append(np.sqrt(2 * quad_coef['a'][ind_tmp - 1]) * (cutpoints[ind_tmp] - cutpoints[ind_tmp - 1]))

        # other intervals on the left
        for i in range(0, ind_tmp - 1):
            # +relu
            temp = 2 * (quad_coef['a'][i] - quad_coef['a'][i + 1]) * cutpoints[i + 1] + (
                    quad_coef['b'][i] - quad_coef['b'][i + 1])
            relu_coef.append(temp)
            relu_intercept.append(-temp * cutpoints[i + 1])

            if quad_coef['a'][i] != 0:
                # +rehu
                rehu_coef.append(-np.sqrt(2 * quad_coef['a'][i]))
                rehu_intercept.append(np.sqrt(2 * quad_coef['a'][i]) * cutpoints[i + 1])
                rehu_cut.append(np.sqrt(2 * quad_coef['a'][i]) * (cutpoints[i + 1] - cutpoints[i]))

        return ReHLoss(np.array(relu_coef), np.array(relu_intercept), np.array(rehu_coef), np.array(rehu_intercept),
                       np.array(rehu_cut))
