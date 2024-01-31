import pytest
import numpy as np
import unittest

from plqcom.PLQLoss import PLQLoss
from plqcom.ReHLoss import ReHLoss

from plqcom.plqutils import PLQProperty


class Test_PLQLoss(unittest.TestCase):
    def setUp(self):
        self.PLQLoss = PLQLoss(cutpoints=np.array([0]),
                               quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})

    def test_continuous(self):
        self.assertEqual(PLQProperty.is_continuous(self.PLQLoss), True)

    def test_convex(self):
        self.assertEqual(PLQProperty.is_convex(self.PLQLoss), True)

    def test_check_cutoff(self):
        self.assertEqual(PLQProperty.check_cutoff(self.PLQLoss), None)

    def test_find_min(self):
        PLQProperty.find_min(self.PLQLoss)
        self.assertEqual(self.PLQLoss.min_val, 1)

    def test_2ReHLoss(self):
        rehloss = self.PLQLoss._2ReHLoss()
        print(rehloss.relu_coef)
        print(rehloss.relu_intercept)
        print(rehloss.rehu_coef)
        print(rehloss.rehu_intercept)
        print(rehloss.rehu_cut)
        # self.assertEqual(self.PLQLoss._2ReHLoss(), None)

# PLQLoss = PLQLoss(cutpoints=np.array([0]),
#                   quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([1, 1])})
# print(PLQLoss._2ReHLoss().relu_coef)
# print(PLQLoss._2ReHLoss().relu_intercept)


# self.assertCountEqual(self.result, self.expected)

# from collections import Counter
#
# def check_equal_without_sort(arr1, arr2):
#     return Counter(arr1) == Counter(arr2)
#
# arr1 = [3,4,5]
# arr2 = [5,3,4]
# print(check_equal_without_sort(arr1,arr2))
