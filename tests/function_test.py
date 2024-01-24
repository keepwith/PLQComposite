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

# PLQLoss = PLQLoss(cutpoints=np.array([0]), quad_coef={'a': np.array([0, 0]), 'b': np.array([-1, 1]), 'c': np.array([0., 0.])})
# print(PLQLoss._2ReHLoss().relu_coef)
# print(PLQLoss._2ReHLoss().relu_intercept)
