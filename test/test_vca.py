import random
import unittest
import cupy as cp
import numpy as np
from global_ import n_seed_
from src.feature_transformations.vca import VCA


class TestVCA(unittest.TestCase):

    def setUp(self):
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)

    def test_fit(self):
        """Tests whether VCA.fit() behaves as intended."""
        for i in range(1, 5):
            vca = VCA(psi=0.1, max_degree=10)

            X_train = cp.random.random((i, 3))
            X_train_transformed, set_train = vca.fit(X_train)
            self.assertTrue(((1/i * cp.linalg.norm(X_train_transformed)**2 <= 0.1).all()), "Polynomials have to vanish.")

    def test_evaluate(self):
        """Tests whether VCA.evaluate() behaves as intended. This also checks whether SetsVCA.apply_V_transformation()
        behaves as intended."""
        for i in range(1, 5):
            m = random.randint(1, 50)
            n = random.randint(1, 50)
            degree = 10
            vca = VCA(psi=0.1, max_degree=degree)

            X_train = cp.random.random((m, n))
            X_train_transformed, set_train = vca.fit(X_train)
            if X_train_transformed is not None:
                X_test_transformed, set_test = vca.evaluate(X_train)
                if not isinstance(X_train_transformed, cp.ndarray):
                    X_train_transformed = cp.array(X_train_transformed.toarray())
                if not isinstance(X_test_transformed, cp.ndarray):
                    X_test_transformed = cp.array(X_test_transformed.toarray())
                self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

                F_train_coeffs = set_train.F_coefficient_vectors
                F_test_coeffs = set_test.F_coefficient_vectors
                self.assertTrue(len(F_train_coeffs) == len(F_test_coeffs),
                                "Lenghts should be identical for identical data.")
                for idx in range(0, len(F_train_coeffs)):
                    F_train_coeffs_cp = F_train_coeffs[idx]
                    F_test_coeffs_cp = F_test_coeffs[idx]
                    self.assertTrue((abs(F_train_coeffs_cp - F_test_coeffs_cp) <= 10e-10).all(),
                                    "Entries should be identical for identical data.")

                V_train_coeffs = set_train.V_coefficient_vectors
                V_test_coeffs = set_test.V_coefficient_vectors
                self.assertTrue(len(V_train_coeffs) == len(V_test_coeffs),
                                "Lenghts should be identical for identical data.")
                for idx in range(0, len(V_train_coeffs)):
                    if V_train_coeffs[idx] is None:
                        self.assertTrue(V_test_coeffs[idx] is None, "Entries should be identical for identical data.")
                    else:
                        V_train_coeffs_cp = V_train_coeffs[idx]
                        V_test_coeffs_cp = V_test_coeffs[idx]
                        self.assertTrue((abs(V_train_coeffs_cp - V_test_coeffs_cp) <= 10e-10).all(),
                                        "Entries should be identical for identical data.")

    def test_evaluate_transformation(self):
        """Tests whether VCA.evaluate_transformation() behaves as intended."""
        vca = VCA(psi=0.1)
        m, n = random.randint(1, 50), random.randint(1, 50)
        X_train = cp.random.random((m, n))
        vca.fit(X_train)
        _, _, _, _, number_of_terms, _ = vca.evaluate_transformation()
        self.assertTrue(number_of_terms == vca.sets_vca.F_to_array().shape[1], "Should be identical.")

        vca = VCA(psi=0.001)
        X_train = cp.array([[1, 2, 3],
                            [4, 5, 6],
                            [1, 5, 6],
                            [4, 1, 0.2],
                            [7, 8, 9]])
        vca.fit(X_train)
        (_, _, _, number_of_polynomials, _, degree) = vca.evaluate_transformation()
        self.assertTrue(number_of_polynomials == vca.sets_vca.V_to_array().shape[1], "Should be identical.")
        self.assertTrue(abs(degree - 2.272727272727273) <= 1e-8, "Degree is wrong.")
