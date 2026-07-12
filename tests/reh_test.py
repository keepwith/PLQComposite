import unittest

import numpy as np
from rehline import ReHLine, plqERM_Ridge, plq_Ridge_Classifier, plq_Ridge_Regressor

from plqcom import PLQLoss, affine_transformation, plq_to_rehloss


def _classification_data(seed=1024, n=1000, d=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    beta = rng.randn(d)
    y = np.sign(X.dot(beta) + rng.randn(n))
    return X, y


def _regression_data(seed=0, n_samples=1000, n_features=5):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    beta = rng.randn(n_features)
    y = np.dot(X, beta) + rng.normal(scale=0.1, size=n_samples)
    return X, y


class TestReHLineIntegration(unittest.TestCase):
    """Integration tests for low-level ReHLine and sklearn-style APIs."""

    def test_svm_builtin_manual_and_plqcom_agree(self):
        X, y = _classification_data()
        C = 0.5

        clf_builtin = plqERM_Ridge(loss={"name": "svm"}, C=C)
        clf_builtin.fit(X=X, y=y)

        n = X.shape[0]
        clf_manual = ReHLine(C=C)
        clf_manual._U = -y.reshape(1, -1)
        clf_manual._V = np.ones((1, n))
        clf_manual.fit(X=X)

        plqloss = PLQLoss(
            quad_coef={
                "a": np.array([0.0, 0.0]),
                "b": np.array([0.0, 1.0]),
                "c": np.array([0.0, 0.0]),
            },
            cutpoints=np.array([0]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss = affine_transformation(rehloss, n=n, c=1, p=-y, q=1)
        clf_plqcom = ReHLine(C=C)
        clf_plqcom._U, clf_plqcom._V = rehloss.relu_coef, rehloss.relu_intercept
        clf_plqcom.fit(X=X)

        plqloss_affine = PLQLoss(
            quad_coef={
                "a": np.array([0.0, 0.0]),
                "b": np.array([-1.0, 0.0]),
                "c": np.array([1.0, 0.0]),
            },
            cutpoints=np.array([1]),
        )
        rehloss_affine = plq_to_rehloss(plqloss_affine)
        rehloss_affine = affine_transformation(
            rehloss_affine, n=n, c=1, y=y, form="classification"
        )
        clf_affine = ReHLine(C=C)
        clf_affine._U, clf_affine._V = (
            rehloss_affine.relu_coef,
            rehloss_affine.relu_intercept,
        )
        clf_affine.fit(X=X)

        for clf in (clf_manual, clf_plqcom, clf_affine):
            np.testing.assert_allclose(clf.coef_, clf_builtin.coef_, rtol=1e-5, atol=1e-5)

        np.testing.assert_array_equal(clf_builtin._U, clf_plqcom._U)
        np.testing.assert_array_equal(clf_builtin._V, clf_plqcom._V)
        np.testing.assert_array_equal(clf_builtin._U, clf_affine._U)
        np.testing.assert_array_equal(clf_builtin._V, clf_affine._V)

    def test_svm_sklearn_classifier_matches_plqcom(self):
        X, y = _classification_data()
        C = 0.5

        plqloss = PLQLoss(
            quad_coef={
                "a": np.array([0.0, 0.0]),
                "b": np.array([0.0, 1.0]),
                "c": np.array([0.0, 0.0]),
            },
            cutpoints=np.array([0]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=-y, q=1)
        clf_plqcom = ReHLine(C=C)
        clf_plqcom._U, clf_plqcom._V = rehloss.relu_coef, rehloss.relu_intercept
        clf_plqcom.fit(X=X)

        clf_sklearn = plq_Ridge_Classifier(
            loss={"name": "svm"}, C=C, fit_intercept=False
        )
        clf_sklearn.fit(X, y)

        np.testing.assert_allclose(
            clf_sklearn.coef_, clf_plqcom.coef_, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            clf_sklearn.decision_function([[0.1, 0.2, 0.3]]),
            clf_plqcom.decision_function([[0.1, 0.2, 0.3]]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_ssvm_builtin_manual_and_plqcom_agree(self):
        X, y = _classification_data()
        C = 0.5

        clf_builtin = plqERM_Ridge(loss={"name": "sSVM"}, C=C)
        clf_builtin.fit(X=X, y=y)

        n = X.shape[0]
        clf_manual = ReHLine(C=C)
        clf_manual._S = -y.reshape(1, -1)
        clf_manual._T = np.ones((1, n))
        clf_manual._Tau = np.ones((1, n))
        clf_manual.fit(X=X)

        plqloss = PLQLoss(
            quad_coef={
                "a": np.array([0.0, 0.5, 0.0]),
                "b": np.array([0.0, 0.0, 1.0]),
                "c": np.array([0.0, 0.0, -0.5]),
            },
            cutpoints=np.array([0.0, 1.0]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss = affine_transformation(rehloss, n=n, c=1, p=-y, q=1)
        clf_plqcom = ReHLine(C=C)
        clf_plqcom._S, clf_plqcom._T, clf_plqcom._Tau = (
            rehloss.rehu_coef,
            rehloss.rehu_intercept,
            rehloss.rehu_cut,
        )
        clf_plqcom.fit(X=X)

        np.testing.assert_allclose(clf_manual.coef_, clf_builtin.coef_, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(clf_plqcom.coef_, clf_builtin.coef_, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(clf_builtin._S, clf_plqcom._S)
        np.testing.assert_array_equal(clf_builtin._T, clf_plqcom._T)
        np.testing.assert_array_equal(clf_builtin._Tau, clf_plqcom._Tau)

    def test_ssvm_sklearn_classifier_matches_plqcom(self):
        X, y = _classification_data()
        C = 0.5

        plqloss = PLQLoss(
            quad_coef={
                "a": np.array([0.0, 0.5, 0.0]),
                "b": np.array([0.0, 0.0, 1.0]),
                "c": np.array([0.0, 0.0, -0.5]),
            },
            cutpoints=np.array([0.0, 1.0]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=-y, q=1)
        clf_plqcom = ReHLine(C=C)
        clf_plqcom._S, clf_plqcom._T, clf_plqcom._Tau = (
            rehloss.rehu_coef,
            rehloss.rehu_intercept,
            rehloss.rehu_cut,
        )
        clf_plqcom.fit(X=X)

        clf_sklearn = plq_Ridge_Classifier(
            loss={"name": "sSVM"}, C=C, fit_intercept=False
        )
        clf_sklearn.fit(X, y)

        np.testing.assert_allclose(
            clf_sklearn.coef_, clf_plqcom.coef_, rtol=1e-5, atol=1e-5
        )

    def test_ridge_regression_affine_forms_agree(self):
        X, y = _regression_data()
        plqloss = PLQLoss(
            quad_coef={"a": np.array([1.0]), "b": np.array([0.0]), "c": np.array([0.0])},
            cutpoints=np.array([]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss_pq = affine_transformation(rehloss, n=X.shape[0], c=1, p=-1, q=y)
        rehloss_form = affine_transformation(
            rehloss, n=X.shape[0], c=1, y=y, form="regression"
        )

        clf_pq = ReHLine(C=1.0)
        clf_pq._Tau, clf_pq._S, clf_pq._T = (
            rehloss_pq.rehu_cut,
            rehloss_pq.rehu_coef,
            rehloss_pq.rehu_intercept,
        )
        clf_pq.fit(X=X)

        clf_form = ReHLine(C=1.0)
        clf_form._Tau, clf_form._S, clf_form._T = (
            rehloss_form.rehu_cut,
            rehloss_form.rehu_coef,
            rehloss_form.rehu_intercept,
        )
        clf_form.fit(X=X)

        np.testing.assert_allclose(clf_pq.coef_, clf_form.coef_, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(clf_pq._Tau, clf_form._Tau)
        np.testing.assert_array_equal(clf_pq._S, clf_form._S)
        np.testing.assert_array_equal(clf_pq._T, clf_form._T)
        np.testing.assert_array_equal(clf_pq._U, clf_form._U)
        np.testing.assert_array_equal(clf_pq._V, clf_form._V)

    def test_mse_sklearn_regressor_matches_plqcom(self):
        X, y = _regression_data()
        plqloss = PLQLoss(
            quad_coef={"a": np.array([1.0]), "b": np.array([0.0]), "c": np.array([0.0])},
            cutpoints=np.array([]),
        )
        rehloss = plq_to_rehloss(plqloss)
        rehloss = affine_transformation(rehloss, n=X.shape[0], c=1, p=-1, q=y)
        clf_plqcom = ReHLine(C=1.0)
        clf_plqcom._Tau, clf_plqcom._S, clf_plqcom._T = (
            rehloss.rehu_cut,
            rehloss.rehu_coef,
            rehloss.rehu_intercept,
        )
        clf_plqcom.fit(X=X)

        clf_sklearn = plq_Ridge_Regressor(
            loss={"name": "MSE"}, C=1, fit_intercept=False
        )
        clf_sklearn.fit(X, y)

        np.testing.assert_allclose(
            clf_sklearn.coef_, clf_plqcom.coef_, rtol=1e-5, atol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
