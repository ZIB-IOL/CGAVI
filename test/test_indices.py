import unittest
import numpy as np
import cupy as cp
from global_ import gpu_memory_
from src.auxiliary_functions.indices import determine_indices_sorted, determine_indices, get_non_zero_indices
from src.gpu.memory_allocation import set_gpu_memory


class TestGetNecessaryColumnIndices(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)

    def test_determine_indices_sorted(self):
        """Tests whether determine_indices_sorted() behaves as intended."""

        border = cp.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 2, 3]])

        obtained = cp.array([[1, 0, 1, 0, 0, 0, 0],
                             [0, 1, 1, 1, 0, 1, 0],
                             [0, 0, 0, 1, 2, 2, 3]])

        self.assertTrue(determine_indices_sorted(border, obtained) == [0, 1, 4, 6], "Wrong indices returned.")

    def test_determine_indices(self):
        """Tests whether determine_indices() behaves as intended."""

        matrix_2 = cp.array([[1, 2, 3, 4, 5, 1, 2, 3, 4],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        matrix_1 = cp.array([[1, 5, 3, 4],
                             [0, 0, 0, 0]])
        self.assertTrue(determine_indices(matrix_1, matrix_2) == [0, 4, 2, 3])

    def test_get_non_zero_indices(self):
        """Tests whether get_non_zero_indices() behaves as intended."""
        for x in [np.array([1, 0, 2]), np.array([[2, 0, 1, 0]]), cp.array([1, 0, 2]), cp.array([[2, 0, 1, 0]])]:
            self.assertTrue([0, 2] == get_non_zero_indices(x), "Wrong non-zero indices returned.")