import unittest
import cupy as cp
import numpy as np

from src.data_sets.preprocessing import min_max_feature_scaling, split_into_classes


class TestPreprocessing(unittest.TestCase):

    def test_min_max_feature_scaling(self):
        """Tests whether min_max_feature_scaling() behaves as intended."""

        X_train = np.array([[1, 2, 3],
                            [0, -5, 6],
                            [0.1, -1, 3]])

        X_test = np.array([[5, 2, 1],
                           [-5, 2, 1]])

        X_train_scaled, X_test_scaled = min_max_feature_scaling(X_train, X_test)
        assert (abs(X_train_scaled - np.array([[1, 1, 0],
                                               [0, 0, 1],
                                               [0.1, 0.57142857,
                                                0]])) <= 10e-5).all(), "min_max_feature_scaling has an error."

        assert (abs(X_test_scaled - np.array([[5, 1, -2 / 3],
                                              [-5, 1,
                                               -2 / 3]])) <= 10e-5).all(), "min_max_feature_scaling has an error."

    def test_split_into_classes(self):
        """Tests whether split_into_classes() behaves as intended."""

        # In cupy
        X = cp.array([[1, 1],
                      [0, 0],
                      [5, 5],
                      [4, 4],
                      [1, 1],
                      [0, 0],
                      [4, 4],
                      [5, 5]])
        y = X[:, 0]

        Xs = split_into_classes(X, y)
        self.assertTrue((Xs[0] == cp.array([[0, 0],
                                            [0, 0]])).all(), "Wrong split.")
        self.assertTrue((Xs[1] == cp.array([[1, 1],
                                            [1, 1]])).all(), "Wrong split.")
        self.assertTrue((Xs[2] == cp.array([[4, 4],
                                            [4, 4]])).all(), "Wrong split.")
        self.assertTrue((Xs[3] == cp.array([[5, 5],
                                            [5, 5]])).all(), "Wrong split.")

        # In numpy
        X = np.array([[1, 1],
                      [0, 0],
                      [5, 5],
                      [4, 4],
                      [1, 1],
                      [0, 0],
                      [4, 4],
                      [5, 5]])
        y = X[:, 0]

        Xs = split_into_classes(X, y)
        self.assertTrue((Xs[0] == np.array([[0, 0],
                                            [0, 0]])).all(), "Wrong split.")
        self.assertTrue((Xs[1] == np.array([[1, 1],
                                            [1, 1]])).all(), "Wrong split.")
        self.assertTrue((Xs[2] == np.array([[4, 4],
                                            [4, 4]])).all(), "Wrong split.")
        self.assertTrue((Xs[3] == np.array([[5, 5],
                                            [5, 5]])).all(), "Wrong split.")
