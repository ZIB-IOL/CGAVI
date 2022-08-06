import cupy as cp
import numpy as np


def determine_indices(matrix_1: cp.ndarray, matrix_2: cp.ndarray):
    """Creates a list of indices such that matrix_2[:, list] = matrix_1.

    This function does for unsorted matrices what determine_indices_sorted() does for sorted matrices.

    Args:
        matrix_1: cp.ndarray
        matrix_2: cp.ndarray

    Return:
        index_list: list
    """
    index_list = []
    for i in range(0, matrix_1.shape[1]):
        for j in range(0, matrix_2.shape[1]):
            if (matrix_1[:, i] == matrix_2[:, j]).all():
                index_list.append(j)
                break
    return index_list


def determine_indices_sorted(matrix_1: cp.ndarray, matrix_2: cp.ndarray):
    """Both matrices are assumed to be sorted. Gets all indices of columns in matrix_2 that are also in matrix_1.

    Args:
        matrix_1: cp.ndarray
        matrix_2: cp.ndarray

    Return:
        index_list: list
    """
    lowest = 0
    index_list = []
    for i in range(0, matrix_1.shape[1]):
        column_border = matrix_1[:, i]
        for j in range(lowest, matrix_2.shape[1]):
            if (column_border == matrix_2[:, j]).all():
                lowest = j
                index_list.append(lowest)
    return index_list


def find_last_non_zero_entries(matrix: cp.ndarray):
    """Finds the last non-zero entry for each column and creates a list of these indices.

    Example:
        matrix = cp.array([[1, 2, 3, 0],
                          [1, 1, 2, 0],
                          [1, 0, 2, 0]])

        Then, find_last_non_zero_entries(matrix) returns [2, 1, 2, -1].
    """
    indices = cp.where(cp.count_nonzero(matrix, axis=0) == 0, -1,
                       (matrix.shape[0] - 1) - cp.argmin(matrix[::-1, :] == 0, axis=0))
    return indices.tolist()


def find_first_non_zero_entries(matrix: cp.ndarray):
    """Finds the first non-zero entry for each column and creates a list of them.

    In case that the column is a zero-vector, the algorithm will return a 0, not a -1.
    Example:
        matrix = cp.array([[1, 0, 3, 0],
                          [1, 1, 2, 0],
                          [1, 0, 2, 0]])

        Then, find_last_non_zero_entries(matrix) returns [2, 1, 2, 0].
    """
    if matrix.shape[0] == 0:
        return None
    return (matrix != 0).argmax(axis=0).tolist()


def get_non_zero_indices(x):
    """Returns the non-zero indices of x."""
    x = x.flatten()
    array_type = type(x)
    assert array_type in [np.ndarray, cp.ndarray], "Requires either a np.ndarray or cp.ndarray."
    if array_type is np.ndarray:
        return np.where(x)[0].tolist()
    elif array_type is cp.ndarray:
        return cp.where(x)[0].tolist()
