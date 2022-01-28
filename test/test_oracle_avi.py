import random
import unittest
import cupy as cp
import numpy as np
from cupyx.scipy.sparse import isspmatrix_csc

from global_ import gpu_memory_, n_seed_
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.oracle_avi import OracleAVI
from src.gpu.memory_allocation import set_gpu_memory


class TestOracleAVI(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)
        self.psi = 0.1
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)

    def test_evaluate(self):
        """Tests whether OracleAVI.evaluate() behaves as intended."""

        pfw_avi = OracleAVI(psi=self.psi, eps=self.psi, lmbda=self.psi)
        m, n = random.randint(1, 50), random.randint(1, 50)
        X_train = cp.random.random((m, n))
        X_train_transformed, set_t = pfw_avi.fit(X_train)
        X_test_transformed, set_v = pfw_avi.evaluate(X_train)
        if not (isinstance(X_train_transformed, cp.ndarray) or isspmatrix_csc(X_train_transformed)):
            X_train_transformed = cp.array(X_train_transformed.toarray())
        if not (isinstance(X_test_transformed, cp.ndarray) or isspmatrix_csc(X_test_transformed)):
            X_test_transformed = cp.array(X_test_transformed.toarray())
        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

        # Degree termination
        X_train = cp.random.random((6, 3))
        pfw_avi = OracleAVI(psi=0.0045, eps=0.0045, lmbda=0.5, maximum_degree=4)
        X_train_transformed, _ = pfw_avi.fit(X_train)
        X_test_transformed, set_v = pfw_avi.evaluate(X_train)
        if not (isinstance(X_train_transformed, cp.ndarray) or isspmatrix_csc(X_train_transformed)):
            X_train_transformed = cp.array(X_train_transformed.toarray())
        if not (isinstance(X_test_transformed, cp.ndarray) or isspmatrix_csc(X_test_transformed)):
            X_test_transformed = cp.array(X_test_transformed.toarray())
        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

        pfw_avi = OracleAVI(psi=100, eps=10, lmbda=0.5)
        m, n = random.randint(1, 50), random.randint(1, 50)
        X_train = cp.random.random((m, n))
        X_train_transformed, set_t = pfw_avi.fit(X_train)
        X_test_transformed, set_v = pfw_avi.evaluate(X_train)

        if not (isinstance(X_train_transformed, cp.ndarray) or isspmatrix_csc(X_train_transformed)):
            X_train_transformed = cp.array(X_train_transformed.toarray())
        if not (isinstance(X_test_transformed, cp.ndarray) or isspmatrix_csc(X_test_transformed)):
            X_test_transformed = cp.array(X_test_transformed.toarray())
        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

    def test_fit(self):
        """Tests whether OracleAVI.fit() behaves as intended."""

        # Regular termination
        pfw_avi = OracleAVI(psi=self.psi, eps=self.psi, lmbda=self.psi)
        m, n = random.randint(1, 50), random.randint(1, 50)
        X_train = cp.random.random((m, n))
        X_train_transformed, set_t = pfw_avi.fit(X_train)
        for col in range(0, set_t.G_evaluations.shape[1]):
            loss = 1 / fd(set_t.G_evaluations[:, col]).shape[0] * cp.linalg.norm(fd(set_t.G_evaluations[:, col])) ** 2
            self.assertTrue(loss < self.psi, "Polynomials are not psi-approximately vanishing.")

    def test_evaluate_sparsity(self):
        """Tests whether OracleAVI.evaluate_sparsity() behaves as intended."""
        pfw_avi = OracleAVI(psi=self.psi, eps=self.psi, lmbda=self.psi)
        m, n = random.randint(1, 50), random.randint(1, 50)
        X_train = cp.random.random((m, n))
        pfw_avi.fit(X_train)
        _, _, _, _, number_of_terms, _ = pfw_avi.evaluate_sparsity()
        self.assertTrue(number_of_terms == pfw_avi.sets_avi.O_array_evaluations.shape[1], "Should be identical.")
