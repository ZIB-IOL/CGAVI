import unittest
import numpy as np
from global_ import n_seed_
from src.auxiliary_functions.auxiliary_functions import fd
from src.data_sets.data_set_creation import fetch_data_set
from src.data_sets.preprocessing import unison_shuffled_copies, train_test_split, min_max_feature_scaling
from src.feature_transformations.avi import AVI
import cupy as cp


class TestAVI(unittest.TestCase):

    def setUp(self):
        cp.random.seed(n_seed_)
        np.random.seed(n_seed_)

    def test_evaluate(self):
        """Tests whether AVI.evaluate() behaves as intended."""

        for border_type in ["gb", "bb"]:
            for tos in ["deglex", "pearson", "rev_pearson"]:
                X, y = fetch_data_set(name='bank')
                X, y = unison_shuffled_copies(X, y)
                X_train, y_train, X_test, y_test = train_test_split(X, y)

                X_train, X_test = min_max_feature_scaling(X_train, X_test)

                psi = 0.1

                tau = 0.1
                avi = AVI(psi=psi, tau=tau, border_type=border_type, term_ordering_strategy=tos)
                X_train_transformed, sets_train = avi.fit(fd(cp.array(X_train)))
                X_test_transformed, set_test = avi.evaluate(cp.array(X_train))
                if not isinstance(X_train_transformed, cp.ndarray):
                    X_train_transformed = cp.array(X_train_transformed.toarray())
                if not isinstance(X_test_transformed, cp.ndarray):
                    X_test_transformed = cp.array(X_test_transformed.toarray())
                self.assertTrue((X_train_transformed - X_test_transformed <= 1e-10).all(), "Should be identical.")

    def test_evaluate_transformation(self):
        """Tests whether AVI.evaluate_transformation() behaves as intended."""
        for border_type in ["gb", "bb"]:
            for tos in ["deglex", "pearson", "rev_pearson"]:
                X_train = cp.array(([0.1, 0.2, 0.1],
                                    [0.2, 0.3, 0.4],
                                    [0.5, 0.6, 0.5],
                                    [0.2, 0.3, 0.1]))

                psi = 0.1

                tau = 0.01
                avi = AVI(psi=psi, tau=tau, border_type=border_type, term_ordering_strategy=tos)
                avi.fit(fd(cp.array(X_train)))
                (total_number_of_zeros, total_number_of_entries, avg_sparsity,  number_of_polynomials, number_of_terms,
                 degree) = avi.evaluate_transformation()

                self.assertTrue(total_number_of_zeros == 0, "Total number of zeros is wrong.")
                self.assertTrue(total_number_of_entries == 6, "Total number of entries is wrong.")
                self.assertTrue(avg_sparsity == 0.0, "Average sparsity is wrong.")
                self.assertTrue(number_of_polynomials == 3, "Number of polynomials is wrong.")
                self.assertTrue(number_of_terms == avi.sets_avi.O_array_evaluations.shape[1], "Number of terms is wrong.")
                self.assertTrue(degree == 1.0, "Degree is wrong.")
