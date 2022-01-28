import unittest

import numpy as np
from cupyx.scipy.sparse import isspmatrix_csc

from global_ import gpu_memory_, n_seed_
from src.auxiliary_functions.auxiliary_functions import fd
from src.data_sets.data_set_creation import fetch_data_set
from src.data_sets.preprocessing import unison_shuffled_copies, train_test_split, min_max_feature_scaling
from src.feature_transformations.avi import AVI
from src.gpu.memory_allocation import set_gpu_memory
import cupy as cp


class TestAVI(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)

    def test_evaluate(self):
        """Tests whether AVI.evaluate() behaves as intended."""

        X, y = fetch_data_set(name='banknote')
        X, y = unison_shuffled_copies(X, y)
        X_train, y_train, X_test, y_test = train_test_split(X, y)

        X_train, X_test = min_max_feature_scaling(X_train, X_test)

        psi = 0.1

        tau = 0.1
        avi = AVI(psi=psi, tau=tau)
        X_train_transformed, sets_train = avi.fit(fd(cp.array(X_train)))
        X_test_transformed, set_test = avi.evaluate(cp.array(X_train))
        if not (isinstance(X_train_transformed, cp.ndarray) or isspmatrix_csc(X_train_transformed)):
            X_train_transformed = cp.array(X_train_transformed.toarray())
        if not (isinstance(X_test_transformed, cp.ndarray) or isspmatrix_csc(X_test_transformed)):
            X_test_transformed = cp.array(X_test_transformed.toarray())
        self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

    def test_evaluate_sparsity(self):
        """Tests whether AVI.evaluate_sparsity() behaves as intended."""
        X_train = cp.array(([0.1, 0.2, 0.1],
                            [0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.5],
                            [0.2, 0.3, 0.1]))

        psi = 0.1

        tau = 0.01
        avi = AVI(psi=psi, tau=tau)
        avi.fit(fd(cp.array(X_train)))
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = avi.evaluate_sparsity()

        self.assertTrue(total_number_of_zeros == 0, "Total number of zeros is wrong.")
        self.assertTrue(total_number_of_entries == 15, "Total number of entries is wrong.")
        self.assertTrue(avg_sparsity == 0.0, "Average sparsity is wrong.")
        self.assertTrue(number_of_polynomials == 4, "Number of polynomials is wrong.")
        self.assertTrue(number_of_terms == avi.sets_avi.O_array_evaluations.shape[1], "Number of terms is wrong.")
        self.assertTrue(degree == 2, "Degree is wrong.")
