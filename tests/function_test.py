import unittest

import numpy as np

from src.PLQLoss import PLQLoss

class PLQTestCase(unittest.TestCase):
    def setUp(self):
        self.PLQLoss = PLQLoss(cutpoints=np.array([0]),
                               quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([0, 0])})

    def test_continuous(self):
        self.assertEqual(self.PLQLoss.is_continuous(), True)

    def test_convex(self):
        self.assertEqual(self.PLQLoss.is_convex(), True)

    def test_check_cutoff(self):
        self.assertEqual(self.PLQLoss.check_cutoff(), None)

    def test_find_min(self):
        self.assertEqual(self.PLQLoss.find_min(), None)

    def test_2ReHLoss(self):
        self.assertEqual(self.PLQLoss._2ReHLoss(), None)


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