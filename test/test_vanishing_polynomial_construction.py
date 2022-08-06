import cupy as cp
import unittest
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.vanishing_polynomial_construction import find_range_null_vca, \
    approximate_onb_algorithm, qr_decomposition, reduced_row_echelon_form_algorithm, \
    stabilized_reduced_row_echelon_form_algorithm, get_info


class TestFindRangeNullVCA(unittest.TestCase):

    def test_find_range_null_vca(self):
        """Tests whether find_range_null_vca() behaves as intended."""

        for run in range(1, 10):
            F = cp.random.random((5, run))
            C = cp.random.random((5, int(2 * run)))
            psi = 0.1

            (V_gboefficient_vectors, V_evaluation_vectors, F_gboefficient_vectors, F_evaluation_vectors
             ) = find_range_null_vca(F, C, psi=psi)

            if isinstance(F_gboefficient_vectors, cp.ndarray):
                for i in range(0, F_gboefficient_vectors.shape[1]):
                    self.assertTrue((abs(fd(cp.hstack((F, C))).dot(F_gboefficient_vectors[:, i])
                                         - F_evaluation_vectors[:, i]) <= 10e-10).all(), "Error among F polynomials.")
                    self.assertTrue((abs(F_evaluation_vectors[:, i]) > psi).any(), "Error among F polynomials.")

            if isinstance(V_gboefficient_vectors, cp.ndarray):
                for i in range(0, V_gboefficient_vectors.shape[1]):
                    self.assertTrue((abs(fd(cp.hstack((F, C))).dot(V_gboefficient_vectors[:, i])
                                         - V_evaluation_vectors[:, i]) <= 10e-10).all(), "Error among V polynomials.")
                    self.assertTrue((1/5*cp.linalg.norm((V_evaluation_vectors[:, i]))**2 <= psi).all(),
                                    "Error among V polynomials.")


class TestApproximateONBAlgorithm(unittest.TestCase):

    def test_approximate_onb_algorithm(self):
        """Tests whether the approximate_onb_algorithm() behaves as intended."""

        matrix = cp.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        mat = approximate_onb_algorithm(matrix, psi=0.1)
        self.assertTrue(mat.tolist() == [[], [], []])

        matrix = cp.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0.05]])

        mat = approximate_onb_algorithm(matrix, psi=0.1)
        self.assertTrue((mat == cp.array([[0],
                                          [0],
                                          [1]])).all())


class TestStabilizedReducedRowEchelonForm(unittest.TestCase):

    def test_qr_decomposition(self):
        """Tests whether the qr_decomposition() behaves as intended."""
        A = cp.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        Q, R = qr_decomposition(A, tau=0.1)
        self.assertTrue((Q == cp.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])).all(), "Wrong matrix returned.")
        self.assertTrue((R == cp.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]])).all(), "Wrong matrix returned.")
        for i in range(10):
            B = cp.random.random((20, 20))

            qr_decomposition(B, tau=0.1)

    def test_reduced_row_echelon_form_algorithm(self):
        """Tests whether reduced_row_echelon_form_algorithm() behaves as intended."""

        A = cp.array([[1, 1, 0],
                      [0, 2, 0]])
        B = reduced_row_echelon_form_algorithm(A)

        self.assertTrue((B == cp.array([[1, 0, 0],
                                        [0, 2, 0]])).all(), "Wrong matrix returned.")

    def test_stabilized_reduced_row_echelon_form(self):
        """Tests whether stabilized_reduced_row_echelon_form() behaves as intended."""
        A = cp.array([[1, 1, 0.2, 0.5, 0.2],
                      [1, 0.2, 0.1, 0.1, 0.2],
                      [0, 1, 0.1, 0.1, 0.2],
                      [0, 0, 0.2, 0, 0.2]])
        B, indices = stabilized_reduced_row_echelon_form_algorithm(A)
        self.assertTrue(indices == [0, 1, 2, 3], "Wrong indices returned.")


class TestGetInfo(unittest.TestCase):

    def test_get_info(self):
        """Tests whether get_info() behaves as intended."""
        A = cp.array([[1, 1],
                      [0, 0],
                      [1, 0],
                      [0, 1]])

        O_terms = cp.array([[0, 1],
                            [0, 0]])

        border_terms = cp.array([[1, 2],
                                 [1, 0]])
        border_evaluations = border_terms

        coefficient_vectors, new_leading_terms, new_O_terms, new_O_evaluations, new_O_indices = get_info(
            O_terms, border_terms, border_evaluations, A)

        self.assertTrue((coefficient_vectors == A).all(), "Wrong coefficient_vectors returned.")
        self.assertTrue((new_leading_terms == cp.array([[1, 2],
                                                        [1, 0]])).all(), "Wrong new_leading_terms returned.")
        self.assertTrue((new_O_terms.shape == (2, 0)), "Wrong new_O_terms returned.")
        self.assertTrue((new_O_evaluations.shape == (2, 0)), "Wrong new_O_evaluations returned.")
        self.assertTrue(new_O_indices == [], "Wrong new_O_indices returned.")
