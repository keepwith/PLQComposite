import numpy as np
import unittest

from plqcom import PLQLoss
from plqcom import plq_to_rehloss, is_continuous, is_convex, check_cutoff, find_min


class TestPLQLoss(unittest.TestCase):
    def setUp(self):
        self.PLQLoss = PLQLoss(cutpoints=np.array([0]),
                               quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})

    def test_continuous(self):
        self.assertEqual(is_continuous(self.PLQLoss), True)

    def test_convex(self):
        self.assertEqual(is_convex(self.PLQLoss), True)

    def test_check_cutoff(self):
        self.assertEqual(check_cutoff(self.PLQLoss), None)

    def test_find_min(self):
        find_min(self.PLQLoss)
        self.assertEqual(self.PLQLoss.min_val, 1)

    def test_2ReHLoss(self):
        print("test 0")
        rehloss = plq_to_rehloss(self.PLQLoss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)

        # y = x^2
        print("test 1")
        plqloss = PLQLoss(cutpoints=np.array([]),
                          quad_coef={'a': np.array([1]), 'b': np.array([0]), 'c': np.array([0])})
        rehloss = plq_to_rehloss(plqloss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)

        #
        print("test 2")
        plqloss = PLQLoss(cutpoints=np.array([0, 1, 2, 3]),
                          quad_coef={'a': np.array([0, 0, 0, 0, 0]), 'b': np.array([0, 1, 2, 3, 4]),
                                     'c': np.array([0, 0, -1, -3, -6])})
        rehloss = plq_to_rehloss(plqloss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)
        print(rehloss.n)

    def test_max2plq(self):
        #
        print("test 3")
        plqloss = PLQLoss(form="max",
                          quad_coef={'a': np.array([0, 0, 0]), 'b': np.array([-1, 0, 1]), 'c': np.array([-1, 0, -1])})
        rehloss = plq_to_rehloss(plqloss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)

        # y=x^2 y=0 y=2x-1 y=-2x-1
        print("test 4")
        plqloss = PLQLoss(form="max",
                          quad_coef={'a': np.array([1, 0, 0, 0]),
                                     'b': np.array([0, 0, 2, -2]),
                                     'c': np.array([0, 0, -1, -1])})
        rehloss = plq_to_rehloss(plqloss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)

        # y=1 y=-x y=x y=1/9x^2-2/3x+1
        print("test 5")
        plqloss = PLQLoss(form="max",
                          quad_coef={'a': np.array([0, 0, 0, 1]),
                                     'b': np.array([0, -1, 1, -6]),
                                     'c': np.array([1, 0, 0, 9])})
        rehloss = plq_to_rehloss(plqloss)
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)
        print(rehloss.n)
