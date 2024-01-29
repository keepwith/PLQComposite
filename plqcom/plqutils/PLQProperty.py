import numpy as np


class PLQProperty:
    """
        This class is used to check the properties of PLQ functions
    """

    def is_continuous(self, plq_loss):
        """
            check whether the input PLQ function is continuous
        :return: True or False
        """
        # check the continuity at cut points from left to right
        for i in range(plq_loss.n_pieces - 1):
            if (plq_loss.quad_coef['a'][i] * plq_loss.cutpoints[i + 1] ** 2 + plq_loss.quad_coef['b'][i] *
                    plq_loss.cutpoints[i + 1] + plq_loss.quad_coef['c'][i] != plq_loss.quad_coef['a'][i + 1] *
                    plq_loss.cutpoints[i + 1] ** 2 + plq_loss.quad_coef['b'][i + 1] * plq_loss.cutpoints[i + 1] +
                    plq_loss.quad_coef['c'][i + 1]):
                return False

        return True

    def is_convex(self, plq_loss):
        """
            check whether the input PLQ function is convex
        ":return: True or False
        """
        # check the second order derivatives
        if min(plq_loss.quad_coef['a']) < 0:
            return False

        # compare the first order derivatives at cut points
        for i in range(plq_loss.n_pieces - 1):
            if (2 * plq_loss.quad_coef['a'][i] * plq_loss.cutpoints[i + 1] + plq_loss.quad_coef['b'][i] >
                    2 * plq_loss.quad_coef['a'][i + 1] * plq_loss.cutpoints[i + 1] + plq_loss.quad_coef['b'][i + 1]):
                return False

        return True

    def check_cutoff(self, plq_loss):
        """
            check whether there exists a cutoff between the knots
        :return:
        """
        # check the cutoff of each piece
        for i in range(plq_loss.n_pieces - 1):
            if plq_loss.quad_coef['a'][i] != 0:  # only will happen when the quadratic term is not zero
                cutpoint = -plq_loss.quad_coef['b'][i] / (2 * plq_loss.quad_coef['a'][i])
                if plq_loss.cutpoints[i] < cutpoint < plq_loss.cutpoints[i + 1]:  # if the cutoff is between the knots
                    # add the cutoff to the knot list and update the coefficients
                    plq_loss.cutpoints.add(cutpoint, i)
                    plq_loss.quad_coef['a'].add(plq_loss.quad_coef['a'][i], i)
                    plq_loss.quad_coef['b'].add(plq_loss.quad_coef['b'][i], i)
                    plq_loss.quad_coef['c'].add(plq_loss.quad_coef['c'][i], i)

    def find_min(self, plq_loss):
        """
            find the minimum knots and value of the PLQ function
        :return:
        """
        # find the minimum value and knot
        out_cut = plq_loss(plq_loss.cutpoints[1:-1])
        plq_loss.min_val = min(out_cut)
        plq_loss.min_knot = np.argmin(out_cut) + 1
        # remove self.min_val from the PLQ function
        plq_loss.quad_coef['c'] = plq_loss.quad_coef['c'] + (-plq_loss.min_val)
