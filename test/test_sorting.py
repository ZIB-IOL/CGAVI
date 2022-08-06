import cupy as cp
import unittest
from src.auxiliary_functions.sorting import sort_by_pivot, deg_lex_sort, compute_degree, get_unique_gbolumns


class TestSorting(unittest.TestCase):

    def setUp(self):
        self.matrix = cp.array([[1, 2, 3, 4, 3],
                                [0, 1, 2, 3, 2],
                                [1, 3, 1, 2, 1]])

        self.matrix_sorted = cp.array([[1, 3, 3, 2, 4],
                                       [0, 2, 2, 1, 3],
                                       [1, 1, 1, 3, 2]])

        self.matrix_sorted_unique = cp.array([[1, 3, 2, 4],
                                              [0, 2, 1, 3],
                                              [1, 1, 3, 2]])

    def test_deg_lex_sort(self):
        """Tests whether deg_lex_sort() and get_same_value_index() behave as intended."""
        matrix_sorted_1, matrix_sorted_2, _ = deg_lex_sort(self.matrix, self.matrix)
        self.assertTrue((self.matrix_sorted == matrix_sorted_1).all(), "Matrix not sorted correctly.")
        self.assertTrue((self.matrix_sorted == matrix_sorted_2).all(), "Matrix not sorted correctly.")
        deg_lex_sort(cp.array([[1]]))

    def test_get_unique_gbolumns(self):
        """Tests whether get_unique_gbolumns() beahves as intended."""
        matrix_unique_1, matrix_unique_2, _ = get_unique_gbolumns(self.matrix, self.matrix)
        self.assertTrue((self.matrix_sorted_unique == matrix_unique_1).all(), "Matrix not sorted correctly.")
        self.assertTrue((self.matrix_sorted_unique == matrix_unique_2).all(), "Matrix not sorted correctly.")
        get_unique_gbolumns(cp.array([[1]]))

    def test_gbompute_degree(self):
        """Tests whether compute_degree() behaves as intended."""
        self.assertTrue(compute_degree(self.matrix) == [2, 6, 6, 9, 6], "Degrees are not computed correctly.")

    def test_sort_by_pivot(self):
        """Tests whether sort_by_pivot() behaves as intended."""
        A = cp.array([[1, 0, 0, 0],
                      [0, 1, 0, 2],
                      [1, 0, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 2]])
        B = sort_by_pivot(A)
        self.assertTrue((B == cp.array([[0, 1, 0, 0],
                                        [1, 0, 0, 2],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 2]])).all(), "Wrongly sorted.")
