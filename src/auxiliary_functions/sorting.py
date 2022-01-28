import cupy as cp

from src.auxiliary_functions.auxiliary_functions import fd
from src.auxiliary_functions.indices import find_last_non_zero_entries


def sort_by_pivot(vectors: cp.ndarray):
    """Sorts vectors by pivot entry index."""

    indices = find_last_non_zero_entries(vectors)
    new_indices = argsort_list(indices)
    vectors = vectors[:, new_indices]
    return vectors


def argsort_list(seq):
    """Returns indices such that the list is sorted.

    Example: argsort_list([2, 1, 3]) = [1, 0, 2].
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


def deg_lex_sort(matrix: cp.ndarray, matrix_2: cp.ndarray = None):
    """
    Sort the columns of matrix degree-lexicographically and matrix_2 accordingly.

    Args:
        matrix: cp.ndarray
            A term matrix which has to be sorted.values
        matrix_2: cp.ndarray, Optional
            Is getting sorted according to the sorting of matrix. (Default is None.)

    Returns:
        matrix: cp.ndarray
            Matrix degree-lexicographically sorted.
        matrix_2: cp.ndarray
            matrix_2 sorted like matrix.
        sorted_list: list
            Sorting list.
    """
    matrix = cp.vstack((matrix, cp.hstack(compute_degree(matrix))))
    sorted_list = cp.lexsort(matrix)
    matrix = matrix[:, sorted_list]
    if matrix_2 is not None:
        matrix_2 = matrix_2[:, sorted_list]
    return matrix[:-1, :], matrix_2, sorted_list


def compute_degree(matrix):
    """
    Computes the sum of the column entries.

    Args:
        matrix: cp.ndarray
            The matrix whose column degrees have to be computed.

    Return:
        degree_list: cp.ndarray
            A list containing the degrees of all the columns.
    """
    m, n = matrix.shape
    degree_list = []
    for i in range(0, n):
        degree_list.append(int(cp.sum(matrix[:, i])))
    return degree_list


def row_sort(matrix: cp.ndarray):
    """Sort by pivot location of columns. Matrix is going to be as upper triangular as possible."""

    first_nonzero_indices = []
    for i in range(0, matrix.shape[0]):
        first_nonzero_indices.append(int(cp.where(matrix[i, :][cp.newaxis, :] != 0)[1][0]))
    matrix = fd(matrix[argsort_list(first_nonzero_indices), :])
    return matrix


def get_unique_columns(matrix: cp.ndarray, matrix_2: cp.ndarray = None):
    """Returns only the unique columns of matrix and matrix_2 sorted."""
    sorted_matrix, sorted_matrix_2, sorted_list = deg_lex_sort(matrix, matrix_2)
    unique_indices = [0]
    for i in range(1, sorted_matrix.shape[1]):
        if (sorted_matrix[:, i - 1] != sorted_matrix[:, i]).any():
            unique_indices.append(i)
    if sorted_list is not None:
        unique_indices = [int(sorted_list[i]) for i in unique_indices]

    if matrix_2 is not None:
        matrix_2 = matrix_2[:, unique_indices]

    return matrix[:, unique_indices], matrix_2, unique_indices
