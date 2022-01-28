import unittest
import cupy as cp
from global_ import gpu_memory_
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.auxiliary_functions_avi import purge, update_coefficient_vectors
from src.gpu.memory_allocation import set_gpu_memory


class TestPurge(unittest.TestCase):

    def setUp(self):
        set_gpu_memory(gpu_memory_)

    def test_purge(self):
        """Tests whether purge() behaves as intended."""

        matrix_2 = cp.array([[1, 2, 3],
                             [0, 0, 1],
                             [1, 1, 1]])
        matrix_1 = cp.array([[1, 1],
                             [0, 1],
                             [0, 1]])
        matrix_2_purged, matrix_2_purged_2, _ = purge(matrix_2, matrix_2, matrix_1)
        self.assertTrue(matrix_2_purged.shape[1] == 0, "Should be empty.")
        self.assertTrue(matrix_2_purged_2.shape[1] == 0, "Should be empty.")

        matrix_2 = cp.array([[1, 2, 3],
                             [0, 0, 1],
                             [1, 1, 1]])
        matrix_1 = cp.array([[1, 1],
                             [0, 2],
                             [2, 1]])
        matrix_2_purged, matrix_2_purged_2, _ = purge(matrix_2, matrix_2, matrix_1)
        self.assertTrue((matrix_2_purged == matrix_2_purged_2).all(), "Should be identical.")
        self.assertTrue((matrix_2_purged == cp.array([[1, 2, 3],
                                                      [0, 0, 1],
                                                      [1, 1, 1]])).all(), "Should be identical.")


class TestUpdateCoefficientVectors(unittest.TestCase):
    def setUp(self):
        set_gpu_memory(gpu_memory_)

    def test_update_coefficient_vectors(self):
        """Tests whether update_coefficient_vectors() behaves as intended."""

        G_coefficient_vectors = cp.array([[1],
                                          [2],
                                          [0]])
        vec = cp.array([[1],
                        [2]])
        G_coefficient_vectors = update_coefficient_vectors(G_coefficient_vectors, vec)
        G_coefficient_vectors = cp.vstack((fd(G_coefficient_vectors),
                                           cp.zeros((1, G_coefficient_vectors.shape[1]))))
        G_coefficient_vectors = update_coefficient_vectors(G_coefficient_vectors, vec)
        G_coefficient_vectors = cp.vstack((fd(G_coefficient_vectors),
                                           cp.zeros((1, G_coefficient_vectors.shape[1]))))
        G_coefficient_vectors = cp.vstack((fd(G_coefficient_vectors),
                                           cp.zeros((1, G_coefficient_vectors.shape[1]))))
        vec = cp.array([[1],
                        [2],
                        [3]])

        G_coefficient_vectors = update_coefficient_vectors(G_coefficient_vectors, vec)

        self.assertTrue((G_coefficient_vectors == cp.array([[1., 1., 1., 1.],
                                                            [2., 0., 0., 0.],
                                                            [0., 2., 0., 0.],
                                                            [0., 0., 2., 0.],
                                                            [0., 0., 0., 2.],
                                                            [0., 0., 0., 3.]])).all(), "Error.")
