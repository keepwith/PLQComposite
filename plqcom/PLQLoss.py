import numpy as np

from plqcom.ReHLoss import ReHLoss


class PLQLoss(object):
    """
        Piecewise Linear Quadratic Loss function
    """

    def __init__(self, quad_coef, type="plq", **paras):
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
            y[cond_tmp] = self.quad_coef['a'][i] * x[cond_tmp] ** 2 + self.quad_coef['b'][i] * x[cond_tmp] + \
                          self.quad_coef['c'][i]

        # add back the minimum value
        if self.min_val != np.inf:
            return y + self.min_val
        else:
            return y

    def minimax2plq(self, quad_coef):
        return quad_coef, np.array([0]), 1

    def _2ReHLoss(self):
        """
            convert the PLQ function to a ReHLoss function
        :return:
            an object of ReHLoss
        """

        # check the continuity and convexity of the PLQ function
        if not self.is_continuous():
            print("The PLQ function is not continuous!")
            exit()

        if not self.is_convex():
            print("The PLQ function is not convex!")
            exit()

        # find the minimum value and knot
        self.find_min()

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
