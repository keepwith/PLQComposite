from plqcom.PLQLoss import PLQLoss
from plqcom.ReHProperty import affine_transformation

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
    rehloss = plqloss._2ReHLoss()
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=-y, q=1)
    clf_3 = ReHLine(loss={'name': 'custom'}, C=C)
    clf_3.U, clf_3.V = rehloss.relu_coef, rehloss.relu_intercept
    clf_3.fit(X=X)
    print('sol privided by rehline: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1.U, clf_3.U))
    print(np.array_equal(clf_1.V, clf_3.V))


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
    rehloss = plqloss._2ReHLoss()
    rehloss = affine_transformation(rehloss, n=X.shape[0], c=C, p=-y, q=1)
    clf_3 = ReHLine(loss={'name': 'custom'}, C=C)
    clf_3.S, clf_3.T, clf_3.Tau = rehloss.rehu_coef, rehloss.rehu_intercept, rehloss.rehu_cut
    clf_3.fit(X=X)
    print('sol privided by rehline: %s' % clf_3.coef_)
    print(clf_3.decision_function([[.1, .2, .3]]))

    print(np.array_equal(clf_1.S, clf_3.S))
    print(np.array_equal(clf_1.T, clf_3.T))
    print(np.array_equal(clf_1.Tau, clf_3.Tau))


if __name__ == '__main__':
    svm_test()
    ssvm_test()
