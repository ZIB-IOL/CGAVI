import random
import unittest
import cupy as cp
import numpy as np
from global_ import n_seed_
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.oracle_avi import OracleAVI


class TestOracleAVI(unittest.TestCase):

    def setUp(self):
        self.psi = 0.1
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)

    def test_evaluate(self):
        """Tests whether OracleAVI.evaluate() behaves as intended. Also tests the setting inverse_hessian_boost."""
        for border_type in ["gb", "bb"]:
            for oracle in ["CG", "PCG", "BPCG", "AGD", "ABM"]:
                for inverse_hessian_boost in ["false", "weak",  "full"]:
                    if inverse_hessian_boost:
                        lmbda = 0
                    else:
                        lmbda = 0.1
                    if (oracle == "ABM" and inverse_hessian_boost is not "false") or\
                            (oracle in ["PCG", "BPCG"] and inverse_hessian_boost == "full") or\
                            (oracle == "AGD" and inverse_hessian_boost == "weak"):
                        pass
                    else:
                        o_avi = OracleAVI(psi=self.psi, eps=self.psi, lmbda=lmbda, border_type=border_type,
                                          inverse_hessian_boost=inverse_hessian_boost, oracle_type=oracle,
                                          region_type="L1Ball")
                        m, n = random.randint(15, 25), random.randint(4, 10)
                        X_train = cp.random.random((m, n))
                        X_train_transformed, set_t = o_avi.fit(X_train)
                        X_test_transformed, set_v = o_avi.evaluate(X_train)
                        if not isinstance(X_train_transformed, cp.ndarray):
                            X_train_transformed = cp.array(X_train_transformed.toarray())
                        if not isinstance(X_test_transformed, cp.ndarray):
                            X_test_transformed = cp.array(X_test_transformed.toarray())
                        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(),
                                        "Should be identical.")

                        # Degree termination
                        X_train = cp.random.random((6, 3))
                        o_avi = OracleAVI(psi=0.0045, eps=0.0045, lmbda=lmbda, max_degree=4, border_type=border_type,
                                          inverse_hessian_boost=inverse_hessian_boost, oracle_type=oracle)
                        X_train_transformed, _ = o_avi.fit(X_train)
                        X_test_transformed, set_v = o_avi.evaluate(X_train)
                        if X_test_transformed is not None:
                            if not isinstance(X_train_transformed, cp.ndarray):
                                X_train_transformed = cp.array(X_train_transformed.toarray())
                            if not isinstance(X_test_transformed, cp.ndarray):
                                X_test_transformed = cp.array(X_test_transformed.toarray())
                            self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(),
                                            "Should be identical.")

                        o_avi = OracleAVI(psi=100, eps=10, lmbda=lmbda, inverse_hessian_boost=inverse_hessian_boost,
                                          border_type=border_type, oracle_type=oracle)
                        m, n = random.randint(15, 25), random.randint(4, 10)
                        X_train = cp.random.random((m, n))
                        X_train_transformed, set_t = o_avi.fit(X_train)
                        X_test_transformed, set_v = o_avi.evaluate(X_train)

                        if not isinstance(X_train_transformed, cp.ndarray):
                            X_train_transformed = cp.array(X_train_transformed.toarray())
                        if not isinstance(X_test_transformed, cp.ndarray):
                            X_test_transformed = cp.array(X_test_transformed.toarray())
                        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(),
                                        "Should be identical.")

    def test_fit(self):
        """Tests whether OracleAVI.fit() behaves as intended."""

        for oracle in ["CG", "PCG", "BPCG", "AGD", "ABM"]:
            # Regular termination
            o_avi = OracleAVI(psi=self.psi, eps=0.01, inverse_hessian_boost="false", oracle_type=oracle)
            m, n = random.randint(15, 25), random.randint(4, 10)
            X_train = cp.random.random((m, n))
            X_train_transformed, set_t = o_avi.fit(X_train)
            for col in range(0, set_t.G_evaluations.shape[1]):
                loss = 1 / fd(set_t.G_evaluations[:, col]).shape[0] * cp.linalg.norm(
                    fd(set_t.G_evaluations[:, col])) ** 2
                self.assertTrue(loss < self.psi, "Polynomials are not psi-approximate vanishing.")

    def test_evaluate_transformation(self):
        """Tests whether OracleAVI.evaluate_transformation() behaves as intended."""
        o_avi = OracleAVI(psi=self.psi, eps=self.psi, lmbda=self.psi, inverse_hessian_boost="false")
        m, n = random.randint(15, 25), random.randint(4, 10)
        X_train = cp.random.random((m, n))
        o_avi.fit(X_train)
        _, _, _, _, number_of_terms, _ = o_avi.evaluate_transformation()
        self.assertTrue(number_of_terms == o_avi.sets_avi.O_array_evaluations.shape[1], "Should be identical.")


