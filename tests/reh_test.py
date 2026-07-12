from plqcom import PLQLoss, plq_to_rehloss, affine_transformation

# test SVM on simulated dataset
import numpy as np
from rehline import ReHLine
from rehline import plqERM_Ridge


def svm_test():
    print("svm test")
    # simulate classification dataset
    n, d, C = 1000, 3, 0.5
    np.random.seed(1024)
    X = np.random.randn(1000, 3)
    beta0 = np.random.randn(3)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Usage 1: build-in loss
    clf_1 = plqERM_Ridge(loss={'name': 'svm'})
    clf_1.C = C
    clf_1.fit(X=X, y=y)
    print('sol provided by rehline: %s' % clf_1.coef_)
    print(clf_1.decision_function([[.1, .2, .3]]))

    # Usage 2: manually specify params (unscaled; C is applied in ReHLine.fit)
    n, d = X.shape
    U = -y.reshape(1, -1)
    V = np.ones((1, n))
    clf_2 = ReHLine(C=C)
    clf_2._U, clf_2._V = U, V
    clf_2.fit(X=X)
    print('sol provided by rehline: %s' % clf_2.coef_)
    print(clf_2.decision_function([[.1, .2, .3]]))

    # Usage 3: manually specify params by PLQComposition Decomposition

    plqloss = PLQLoss(quad_coef={'a': np.array([0., 0.]), 'b': np.array([0., 1.]), 'c': np.array([0., 0.])},
                      cutpoints=np.array([0]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=-y, q=1)
    clf_3 = ReHLine(C=C)
    clf_3._U, clf_3._V = rehloss.relu_coef, rehloss.relu_intercept
    clf_3.fit(X=X)
    print('sol provided by rehline with form custom: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1._U, clf_3._U))
    print(np.array_equal(clf_1._V, clf_3._V))

    # Usage 4: manually specify params by PLQComposition Decomposition and form classifcation form affine transformation
    plqloss = PLQLoss(quad_coef={'a': np.array([0., 0.]), 'b': np.array([-1., 0.]), 'c': np.array([1., 0.])},
                      cutpoints=np.array([1]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, y=y, form='classification')
    clf_4 = ReHLine(C=C)
    clf_4._U, clf_4._V = rehloss.relu_coef, rehloss.relu_intercept
    clf_4.fit(X=X)
    print('sol provided by rehline with form affine: %s' % clf_4.coef_)
    print(clf_4.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1._U, clf_3._U))
    print(np.array_equal(clf_1._V, clf_3._V))
    print(np.array_equal(clf_1._U, clf_4._U))
    print(np.array_equal(clf_1._V, clf_4._V))


def ssvm_test():
    print("ssvm test")
    # simulate classification dataset
    n, d, C = 1000, 3, 0.5
    np.random.seed(1024)
    X = np.random.randn(1000, 3)
    beta0 = np.random.randn(3)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Usage 1: build-in loss
    clf_1 =plqERM_Ridge(loss={'name': 'sSVM'}, C=C)
    clf_1.fit(X=X, y=y)
    print('sol provided by rehline: %s' % clf_1.coef_)
    print(clf_1.decision_function([[.1, .2, .3]]))

    # Usage 2: manually specify params (unscaled; C is applied in ReHLine.fit)
    n, d = X.shape
    S = -y.reshape(1, -1)
    T = np.ones((1, n))
    Tau = np.ones((1, n))
    clf_2 = ReHLine(C=C)
    clf_2._S, clf_2._T, clf_2._Tau = S, T, Tau
    clf_2.fit(X=X)
    print('sol provided by rehline: %s' % clf_2.coef_)
    print(clf_2.decision_function([[.1, .2, .3]]))

    # Usage 3: manually specify params by PLQComposition Decomposition

    plqloss = PLQLoss(
        quad_coef={'a': np.array([0., 0.5, 0.]), 'b': np.array([0., 0., 1]), 'c': np.array([0., 0., -0.5])},
        cutpoints=np.array([0., 1.]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=-y, q=1)
    clf_3 = ReHLine(C=C)
    clf_3._S, clf_3._T, clf_3._Tau = rehloss.rehu_coef, rehloss.rehu_intercept, rehloss.rehu_cut
    clf_3.fit(X=X)
    print('sol provided by rehline: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1._S, clf_3._S))
    print(np.array_equal(clf_1._T, clf_3._T))
    print(np.array_equal(clf_1._Tau, clf_3._Tau))


def ridge_regression_test():
    n_samples, n_features = 1000, 5
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features)
    beta = rng.randn(n_features)
    y = np.dot(X, beta) + rng.normal(scale=0.1, size=n_samples)
    plqloss = PLQLoss(quad_coef={'a': np.array([1.]), 'b': np.array([0.]), 'c': np.array([0.])},
                      cutpoints=np.array([]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss.rehu_cut, rehloss.rehu_coef, rehloss.rehu_intercept
    rehloss_1 = affine_transformation(rehloss, n=X.shape[0], c=1, p=-1, q=y)
    rehloss_2 = affine_transformation(rehloss, n=X.shape[0], c=1, y=y, form='regression')
    clf_1 = ReHLine(C=1.0)
    clf_1._Tau, clf_1._S, clf_1._T = rehloss_1.rehu_cut, rehloss_1.rehu_coef, rehloss_1.rehu_intercept
    clf_1.fit(X=X)
    print('sol provided by rehline: %s' % clf_1.coef_)

    clf_2 = ReHLine(C=1.0)
    clf_2._Tau, clf_2._S, clf_2._T = rehloss_2.rehu_cut, rehloss_2.rehu_coef, rehloss_2.rehu_intercept
    clf_2.fit(X=X)
    print('sol provided by rehline: %s' % clf_2.coef_)
    print(np.array_equal(clf_1._Tau, clf_2._Tau))
    print(np.array_equal(clf_1._S, clf_2._S))
    print(np.array_equal(clf_1._T, clf_2._T))
    print(np.array_equal(clf_1._U, clf_2._U))
    print(np.array_equal(clf_1._V, clf_2._V))


if __name__ == '__main__':
    svm_test()
    ssvm_test()
    ridge_regression_test()
