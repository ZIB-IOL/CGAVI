import unittest
import numpy as np
import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd, orthogonal_projection


class TestAuxiliaryFunctions(unittest.TestCase):

    def test_fd(self):
        """Tests whether fd() behaves as intended."""

        self.assertTrue(fd(cp.random.rand(5, 1)).shape == (5, 1), "Error for np.array of 1 dimensions.")
        self.assertTrue(fd(cp.random.rand(1, 5).flatten()).shape == (5, 1), "Error for np.array of 1 dimensions.")
        self.assertTrue(fd(cp.random.rand(5, 1)).shape == (5, 1), "Error for cp.array of 1 dimensions.")
        self.assertTrue(fd(cp.random.rand(1, 5).flatten()).shape == (5, 1), "Error for cp.array of 1 dimensions.")
        self.assertTrue(fd(np.random.rand(5, 2)).shape == (5, 2), "Error for np.array of 2 dimensions.")
        self.assertTrue(fd(cp.random.rand(5, 2)).shape == (5, 2), "Error for cp.array of 2 dimensions.")

    def test_orthogonal_projection(self):
        """Tests whether orthogonal_projection() behaves as intended."""
        A = cp.array([[1, 0.5],
                      [0, 1]])
        b = cp.array([[1],
                      [1]])

        orthogonal_gbomponents = orthogonal_projection(A, b)
        self.assertTrue((orthogonal_gbomponents == cp.array([[1], [1.5]])).all(),
                        "Returned orthogonal_gbomponents are incorrect.")


