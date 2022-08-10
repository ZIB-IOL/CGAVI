import unittest
import cupy as cp
import numpy as np
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.auxiliary_functions_avi import purge, update_coefficient_vectors, \
    streaming_matrix_updates


class TestPurge(unittest.TestCase):

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

    def test_streaming_matrix_updates(self):
        """Tests whether streaming_matrix_updates() behaves as intended."""

        dim = 30000
        A = cp.random.rand(dim, 500)
        A_squared = A.T.dot(A)
        A_squared_inv = cp.asnumpy(A_squared)
        A_squared_inv = cp.array(np.linalg.inv(A_squared_inv))

        a = cp.random.rand(dim, 1)
        a_squared = float(a.T.dot(a))

        A_a = A.T.dot(a)

        B, B_2, B_2_1 = streaming_matrix_updates(A, A_squared, A_a, a, a_squared, A_squared_inv=A_squared_inv)

        C = cp.hstack((A, a))
        C_2 = C.T.dot(C)

        C_2_1 = cp.asnumpy(C_2)
        C_2_1 = cp.array(np.linalg.inv(C_2_1))

        self.assertAlmostEqual(float(cp.linalg.norm((C_2 - B_2))), 0, 8,
                               "The function streaming_matrix_updates does not work as intended.")

        self.assertAlmostEqual(float(cp.linalg.norm((C_2_1 - B_2_1))), 0, 8,
                               "The function streaming_matrix_updates does not work as intended.")



