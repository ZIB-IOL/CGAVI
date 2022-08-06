import unittest
import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.terms_and_polynomials import SetsOAndG, SetsVCA


class TestSetsOAndG(unittest.TestCase):

    def test_evaluate_transformation(self):
        """Tests whether SetsOAndG.evaluate_transformation() beahves as intended."""
        X_train = cp.array([[1, 2],
                            [3, 4],
                            [5, 6]])

        sets_avi = SetsOAndG(X_train)

        sets_avi.G_gboefficient_vectors = [None, None, cp.array([[1, 0],
                                                                [2, 1],
                                                                [3, 0],
                                                                [0, 3]]), None, cp.array([[1, 1, 1, 1],
                                                                                          [2, 0, 0, 0],
                                                                                          [0, 1, 0, 0],
                                                                                          [0, 0, 2, 0],
                                                                                          [0, 0, 1, 0],
                                                                                          [0, 0, 0, 1]])]

        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = sets_avi.evaluate_transformation()
        self.assertTrue(total_number_of_zeros == 2, "Error.")
        self.assertTrue(total_number_of_entries == 16, "Error.")
        self.assertTrue((abs(avg_sparsity - 0.16666666666666666)) < 1e-8, "Error.")
        self.assertTrue(number_of_terms == sets_avi.O_array_evaluations.shape[1], "Error.")
        self.assertTrue(number_of_polynomials == 6, "Error.")
        self.assertTrue((abs(degree - 3.3333333333333335)) < 1e-8, "Error.")


class TestSetsVCA(unittest.TestCase):

    def test_everything(self):
        """Tests whether SetsVCA.update_F(), SetsVCA.F_to_array(), SetsVCA.update_V(), SetsVCA.V_to_array(),
        SetsVCA.update_C(), SetsVCA.construct_border(), and SetsVCA.evaluate_transformation() behave as intended."""

        X = cp.array([[0.1, 0.4],
                      [0.2, 0.5],
                      [0.3, 0.6]])
        sets_vca = SetsVCA(X)

        self.assertTrue((sets_vca.X == fd(X)).all(), "Should be identical.")
        self.assertTrue(sets_vca.Cs == [], "Should be identical.")
        self.assertTrue(sets_vca.Vs == [], "Should be identical.")
        self.assertTrue(sets_vca.V_gboefficient_vectors == [], "Should be identical.")
        self.assertTrue((sets_vca.Fs[0] == fd(cp.ones((X.shape[0], 1)) / cp.sqrt(X.shape[0]))).all(),
                        "Should be identical.")
        self.assertTrue((sets_vca.F_gboefficient_vectors[0] == cp.array([[1]])).all(), "Should be identical.")

        border = sets_vca.construct_border()
        self.assertTrue((border == X).all(), "Should be identical.")

        F_gboefficient_vectors = cp.array([[1, 0.5],
                                          [0.1, 0.2]])
        F_evaluation_vectors = cp.array([[0.5, 2],
                                         [0.3, 1],
                                         [0.4, 1]])
        sets_vca.update_F(F_gboefficient_vectors, F_evaluation_vectors)
        self.assertTrue((sets_vca.Fs[1] == fd(F_evaluation_vectors)).all(), "Should be identical.")
        self.assertTrue((sets_vca.F_gboefficient_vectors[1] == fd(F_gboefficient_vectors)).all(), "Should be identical.")

        V_gboefficient_vectors = cp.array([[0.5]])
        V_evaluation_vectors = cp.array([[0.1],
                                         [0.1],
                                         [0.2]]).flatten()
        sets_vca.update_V(V_gboefficient_vectors, V_evaluation_vectors)
        self.assertTrue((sets_vca.Vs[0] == fd(V_evaluation_vectors)).all(), "Should be identical.")
        self.assertTrue((sets_vca.V_gboefficient_vectors[0] == fd(V_gboefficient_vectors)).all(), "Should be identical.")

        sets_vca.update_V(None, None)
        self.assertTrue(sets_vca.Vs[1] is None, "Should be None.")
        self.assertTrue(sets_vca.V_gboefficient_vectors[1] is None, "Should be None.")

        V_gboefficient_vectors = cp.array([[0.5],
                                          [0.2],
                                          [0]])
        V_evaluation_vectors = cp.array([[-0.3],
                                         [0.0],
                                         [0.2]]).flatten()
        sets_vca.update_V(V_gboefficient_vectors, V_evaluation_vectors)
        self.assertTrue((sets_vca.Vs[2] == fd(V_evaluation_vectors)).all(), "Should be identical.")
        self.assertTrue((sets_vca.V_gboefficient_vectors[2] == fd(V_gboefficient_vectors)).all(), "Should be identical.")

        C_evaluation_vectors = cp.array([[0.1],
                                         [0.2],
                                         [0.3]]).flatten()
        sets_vca.update_C(C_evaluation_vectors)
        self.assertTrue((sets_vca.Cs[0] == fd(C_evaluation_vectors)).all(), "Should be identical.")

        F_gboefficient_vectors = cp.array([[1],
                                          [-1]])
        F_evaluation_vectors = cp.array([[-0.5, 0.6],
                                         [0.3, 0.9],
                                         [0.4, 0.8]])
        sets_vca.update_F(F_gboefficient_vectors, F_evaluation_vectors)

        F_array = sets_vca.F_to_array()
        self.assertTrue((abs(F_array - cp.array([[0.57735027, 0.5, 2, -0.5, 0.6],
                                                 [0.57735027, 0.3, 1, 0.3, 0.9],
                                                 [0.57735027, 0.4, 1, 0.4, 0.8]])) <= 10e-10).all(),
                        "Should be identical.")

        V_array = sets_vca.V_to_array()
        self.assertTrue((V_array == cp.array([[0.1, -0.3],
                                              [0.1, 0],
                                              [0.2, 0.2]])).all(), "Should be identical.")

        (zeros, entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = sets_vca.evaluate_transformation()
        self.assertTrue(zeros == 1, "Number of zeros is wrong.")
        self.assertTrue(entries == 4, "Number of entries is wrong.")
        self.assertTrue(abs(avg_sparsity - 0.25) <= 1e-8, "Average sparsity is wrong.")
        self.assertTrue(number_of_polynomials == 2, "Number of polynomials is wrong.")
        self.assertTrue(number_of_terms == sets_vca.F_to_array().shape[1], "Number of polynomials is wrong.")
        self.assertTrue(abs(degree - 2.0) <= 1e-8, "Degree is wrong.")


 # TODO: Write test for border creation.
