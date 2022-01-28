import os
import sys

import cupy as cp
import numpy as np


def fd(x):
    """Guarantees that x is 2 dimensional."""
    array_type = type(x)
    if x.ndim == 1:
        if array_type is np.ndarray:
            x = x.flatten()[:, np.newaxis]
        elif array_type is cp.ndarray:
            x = x.flatten()[:, cp.newaxis]
    return x


def orthogonal_projection(vectors: cp.ndarray, vector: cp.ndarray):
    """
    Obtain orthogonal components of vector projected onto vectors.

    Args:
        vectors: cp.ndarray
        vector: cp.ndarray

    Returns:
        orthogonal_components: cp.ndarray
            A column vector of the form (vector.T.dot(vectors)).T.
    """

    vector = fd(vector)
    vectors = fd(vectors)
    assert vectors.shape[0] == vector.shape[0], "Vectors have to have identical length."

    orthogonal_components = (vector.T.dot(vectors)).T

    return fd(orthogonal_components)


def evaluate_vanishing(evaluation_matrix: cp.ndarray):
    """Get average level of vanishing, that is, for each column in """
    avg_mse = 0
    for i in range(0, evaluation_matrix.shape[1]):
        avg_mse += cp.linalg.norm(evaluation_matrix[:, i]) ** 2 / evaluation_matrix.shape[0]
    avg_mse = avg_mse / evaluation_matrix.shape[1]
    return avg_mse


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



