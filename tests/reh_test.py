from plqcom import PLQLoss, plq_to_rehloss, affine_transformation

# test SVM on simulated dataset
import numpy as np
from rehline import ReHLine


def svm_test():
    print("svm test")
    # simulate classification dataset
    n, d, C = 1000, 3, 0.5
    np.random.seed(1024)
    X = np.random.randn(1000, 3)
    beta0 = np.random.randn(3)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Usage 1: build-in loss
    clf_1 = ReHLine(loss={'name': 'svm'}, C=C)
    clf_1.make_ReLHLoss(X=X, y=y, loss={'name': 'svm'})
    clf_1.fit(X=X)
    print('sol privided by rehline: %s' % clf_1.coef_)
    print(clf_1.decision_function([[.1, .2, .3]]))

    # Usage 2: manually specify params
    n, d = X.shape
    U = -(C * y).reshape(1, -1)
    L = U.shape[0]
    V = (C * np.array(np.ones(n))).reshape(1, -1)
    clf_2 = ReHLine(loss={'name': 'svm'}, C=C)
    clf_2.U, clf_2.V = U, V
    clf_2.fit(X=X)
    print('sol privided by rehline: %s' % clf_2.coef_)
    print(clf_2.decision_function([[.1, .2, .3]]))

    # Usage 3: manually specify params by PLQComposition Decomposition

    plqloss = PLQLoss(quad_coef={'a': np.array([0., 0.]), 'b': np.array([0., 1.]), 'c': np.array([0., 0.])},
                      cutpoints=np.array([0]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=-y, q=1)
    clf_3 = ReHLine(loss={'name': 'custom'}, C=C)
    clf_3.U, clf_3.V = rehloss.relu_coef, rehloss.relu_intercept
    clf_3.fit(X=X)
    print('sol privided by rehline with form custom: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1.U, clf_3.U))
    print(np.array_equal(clf_1.V, clf_3.V))

    # Usage 4: manually specify params by PLQComposition Decomposition and form classifcation form affine transformation
    plqloss = PLQLoss(quad_coef={'a': np.array([0., 0.]), 'b': np.array([-1., 0.]), 'c': np.array([1., 0.])},
                      cutpoints=np.array([1]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, y=y, form='classification')
    clf_4 = ReHLine(loss={'name': 'custom'}, C=C)
    clf_4.U, clf_4.V = rehloss.relu_coef, rehloss.relu_intercept
    clf_4.fit(X=X)
    print('sol privided by rehline with form affine: %s' % clf_4.coef_)
    print(clf_4.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1.U, clf_3.U))
    print(np.array_equal(clf_1.V, clf_3.V))
    print(np.array_equal(clf_1.U, clf_4.U))
    print(np.array_equal(clf_1.V, clf_4.V))


def ssvm_test():
    print("ssvm test")
    # simulate classification dataset
    n, d, C = 1000, 3, 0.5
    np.random.seed(1024)
    X = np.random.randn(1000, 3)
    beta0 = np.random.randn(3)
    y = np.sign(X.dot(beta0) + np.random.randn(n))

    # Usage 1: build-in loss
    clf_1 = ReHLine(loss={'name': 'sSVM'}, C=C)
    clf_1.make_ReLHLoss(X=X, y=y, loss={'name': 'sSVM'})
    clf_1.fit(X=X)
    print('sol privided by rehline: %s' % clf_1.coef_)
    print(clf_1.decision_function([[.1, .2, .3]]))

    # Usage 2: manually specify params
    n, d = X.shape
    S = -(np.sqrt(C) * y).reshape(1, -1)
    H = S.shape[0]
    T = (np.sqrt(C) * np.array(np.ones(n))).reshape(1, -1)
    Tau = (np.sqrt(C) * np.array(np.ones(n))).reshape(1, -1)
    clf_2 = ReHLine(loss={'name': 'sSVM'}, C=C)
    clf_2.S, clf_2.T, clf_2.Tau = S, T, Tau
    clf_2.fit(X=X)
    print('sol privided by rehline: %s' % clf_2.coef_)
    print(clf_2.decision_function([[.1, .2, .3]]))

    # Usage 3: manually specify params by PLQComposition Decomposition

    plqloss = PLQLoss(
        quad_coef={'a': np.array([0., 0.5, 0.]), 'b': np.array([0., 0., 1]), 'c': np.array([0., 0., -0.5])},
        cutpoints=np.array([0., 1.]))
    rehloss = plq_to_rehloss(plqloss)
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=-y, q=1)
    clf_3 = ReHLine(loss={'name': 'custom'}, C=C)
    clf_3.S, clf_3.T, clf_3.Tau = rehloss.rehu_coef, rehloss.rehu_intercept, rehloss.rehu_cut
    clf_3.fit(X=X)
    print('sol privided by rehline: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1.S, clf_3.S))
    print(np.array_equal(clf_1.T, clf_3.T))
    print(np.array_equal(clf_1.Tau, clf_3.Tau))


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
    clf_1 = ReHLine(loss={'name': 'custom'}, C=1)
    clf_1.Tau, clf_1.S, clf_1.T = rehloss_1.rehu_cut, rehloss_1.rehu_coef, rehloss_1.rehu_intercept
    clf_1.fit(X=X)
    print('sol privided by rehline: %s' % clf_1.coef_)

    clf_2 = ReHLine(loss={'name': 'custom'}, C=1)
    clf_2.Tau, clf_2.S, clf_2.T = rehloss_2.rehu_cut, rehloss_2.rehu_coef, rehloss_2.rehu_intercept
    clf_2.fit(X=X)
    print('sol privided by rehline: %s' % clf_2.coef_)
    print(np.array_equal(clf_1.Tau, clf_2.Tau))
    print(np.array_equal(clf_1.S, clf_2.S))
    print(np.array_equal(clf_1.T, clf_2.T))
    print(np.array_equal(clf_1.U, clf_2.U))
    print(np.array_equal(clf_1.V, clf_2.V))


if __name__ == '__main__':
    svm_test()
    ssvm_test()
    ridge_regression_test()
