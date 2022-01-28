import unittest
import numpy as np
import cupy as cp

from src.auxiliary_functions.auxiliary_functions import fd, orthogonal_projection, evaluate_vanishing
from src.gpu.memory_allocation import set_gpu_memory

from global_ import gpu_memory_


class TestAuxiliaryFunctions(unittest.TestCase):
    def setUp(self):
        set_gpu_memory(gpu_memory_)

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

        orthogonal_components = orthogonal_projection(A, b)
        self.assertTrue((orthogonal_components == cp.array([[1], [1.5]])).all(),
                        "Returned orthogonal_components are incorrect.")

    def test_evaluate_vanishing(self):
        """Tests whether evaluate_vanishing() behaves as intended."""
        X_train = cp.array([[1, 1, 1],
                           [0, 0, 0]])
        avg_mse = evaluate_vanishing(X_train)
        self.assertTrue(avg_mse == 0.5, "Wrong evaluation.")

        X_train = cp.array([[1, 1, 1],
                           [1, 0, 0]])
        avg_mse = evaluate_vanishing(X_train)
        self.assertTrue(avg_mse == 2 / 3, "Wrong evaluation.")