import unittest
import cupy as cp
import random
from src.feature_transformations.term_ordering_strategies import pearson


class TestTermOrderingStrategies(unittest.TestCase):

    def test_pearson(self):
        """Tests whether pearson() behaves as intended."""

        A = cp.array([[-1, 2, -.05],
                      [-1, 3, -1],
                      [-1, 2, -3],
                      [-3, 4, 0.1]])

        term_ordering = pearson(A)
        self.assertTrue(term_ordering == [2, 0, 1], "The function pearson() is not working properly.")

        term_ordering_2 = pearson(A, rev=True)
        self.assertTrue(term_ordering_2 == [1, 0, 2], "The function pearson() is not working properly.")

        self.assertTrue(term_ordering != term_ordering_2, "The function pearson() is not working properly.")

        A = cp.random.rand(5, 3)
        term_ordering = pearson(A)
        sol_1 = A[:, term_ordering]

        indices = list(range(A.shape[1]))
        random.shuffle(indices)
        A = A[:, indices]
        term_ordering_2 = pearson(A)
        sol_2 = A[:, term_ordering_2]

        self.assertTrue((sol_1 == sol_2).all(), "The function pearson() cannot depend on the order of the features.")
