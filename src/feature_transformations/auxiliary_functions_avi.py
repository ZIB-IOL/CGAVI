from itertools import compress
import cupy as cp
import numpy as np
from src.auxiliary_functions.auxiliary_functions import fd
from src.auxiliary_functions.indices import find_last_non_zero_entries, find_first_non_zero_entries
from src.auxiliary_functions.sorting import get_unique_gbolumns


def construct_border(terms: cp.ndarray, terms_evaluated: cp.ndarray, X_train: cp.ndarray,
                     degree_1_terms: cp.ndarray = None, degree_1_terms_evaluated: cp.ndarray = None,
                     purging_terms: cp.ndarray = None):
    """Constructs the border of terms.

    Args:
        terms: cp.ndarray
        terms_evaluated: cp.ndarray
        X_train: cp.ndarray
        degree_1_terms: cp.ndarray, Optional
            (Default is None.)
        degree_1_terms_evaluated: cp.ndarray, Optional
            (Default is None.)
        purging_terms: cp.ndarray, Optional
            Purge all terms divisible by these terms. (Default is None.)

    Returns:
        border_terms_raw: cp.ndarray
        border_evaluations_raw: cp.ndarray
        non_purging_indices: list
    """
    terms = fd(terms)
    terms_evaluated = fd(terms_evaluated)
    X_train = fd(X_train)

    if degree_1_terms is None:
        border_terms_raw, border_evaluations_raw = cp.identity(X_train.shape[1]), X_train
    else:
        terms = fd(terms)
        terms_evaluated = fd(terms_evaluated)

        degree_1_terms = fd(degree_1_terms)
        degree_1_terms_evaluated = fd(degree_1_terms_evaluated)

        terms_repeat = cp.repeat(terms, repeats=degree_1_terms.shape[1], axis=1)
        degree_1_terms_tile = cp.tile(degree_1_terms, (1, terms.shape[1]))
        border_terms_raw = fd(degree_1_terms_tile + terms_repeat)

        terms_evaluated_repeat = cp.repeat(terms_evaluated, repeats=degree_1_terms_evaluated.shape[1], axis=1)
        degree_1_terms_evaluated_tile = cp.tile(degree_1_terms_evaluated, (1, terms_evaluated.shape[1]))
        border_evaluations_raw = cp.multiply(degree_1_terms_evaluated_tile, terms_evaluated_repeat)

    border_terms_purged, border_evaluations_purged, unique_indices = get_unique_gbolumns(
        border_terms_raw, border_evaluations_raw)

    if purging_terms is not None:
        border_terms_purged, border_evaluations_purged, unique_indices_2 = purge(
            border_terms_purged, border_evaluations_purged, purging_terms)
        if unique_indices_2 is not None:
            non_purging_indices = [unique_indices[i] for i in unique_indices_2]
        else:
            non_purging_indices = unique_indices
    else:
        non_purging_indices = unique_indices

    return border_terms_raw, border_evaluations_raw, non_purging_indices


def purge(terms: cp.ndarray, terms_evaluated: cp.ndarray, purging_terms: cp.ndarray):
    """Purges the purging_terms from terms and updates terms_evaluated accordingly.

    If any term in terms is a power of a term in purging_terms, it gets deleted. The corresponding column in
    terms_evaluated gets deleted as well.
    """
    indices = [x for x in range(0, terms.shape[1])]
    for i in range(0, purging_terms.shape[1]):
        illegal = fd(purging_terms[:, i])
        list_keep = cp.any(terms[:, indices] - illegal < 0, axis=0).tolist()
        indices = list(compress(indices, list_keep))
    return terms[:, indices], terms_evaluated[:, indices], indices


def update_gboefficient_vectors(G_gboefficient_vectors: cp.ndarray, coefficient_vector: cp.ndarray, first: bool = False):
    """Appends a polynomial with coefficient vector based on coefficient_vector and term to G_gboefficient_vectors."""
    if G_gboefficient_vectors is None:
        G_gboefficient_vectors = fd(coefficient_vector)
    else:
        if first:
            lt_indices = find_first_non_zero_entries(G_gboefficient_vectors)
        else:
            lt_indices = find_last_non_zero_entries(G_gboefficient_vectors)
        removable_set = set(list(lt_indices))
        indices = [x for x in list(range(0, G_gboefficient_vectors.shape[0])) if x not in removable_set]
        if len(indices) == fd(coefficient_vector).shape[0]:
            updated_gboefficient_vector = cp.zeros((G_gboefficient_vectors.shape[0], 1))
            updated_gboefficient_vector[indices, :] = fd(coefficient_vector)
            G_gboefficient_vectors = cp.hstack((G_gboefficient_vectors, updated_gboefficient_vector))
    return G_gboefficient_vectors


def streaming_matrix_updates(A: cp.ndarray, A_squared: cp.ndarray, A_a: cp.ndarray, a: cp.ndarray, a_squared: float,
                             A_squared_inv: cp.ndarray = None, built_in: bool = False):
    """Given a matrix A, A.T.A, and (A.T.A)^-1, we want to efficiently compute B =[A, a], B.T.B, and (B.T.B)^-1. This is
    necessary for fast Inverse Hessian Boosting.

    Args:
        A: cp.ndarray
        A_squared: cp.ndarray
        A_a: cp.ndarray
        a: cp.ndarray
        a_squared: cp.ndarray
        A_squared_inv: cp.ndarray, Optional
            (Default is None.)
        built_in: bool, Optional
            Whether to invert B.T.B using the built-in matrix inversion algorithm (True) or with the efficient update
            strategy (False).
             (Default is False.)

    If interested in (B.T.B)^-1, the matrices A.T.A and B.T.B have to be invertible. This is guaranteed for ABM and
    OAVI-type algorithms that run with AGD.
    """
    B_squared_inv = None

    if built_in:
        B = cp.hstack((A, a))
        B_squared = B.T.dot(B)
        if A_squared_inv is not None:
            B_squared_inv = cp.array(np.linalg.inv(B_squared))
    else:
        # B
        B = cp.hstack((A, a))

        b = A_a

        # B_squared
        B_squared = cp.hstack((A_squared, b))
        B_squared = cp.vstack((B_squared, cp.vstack((b, a_squared)).T))

        # B_squared_inv
        if A_squared_inv is not None:
            # We write B_squared_inv as S = | S_1   s_2 |
            #                               | s_2.T s_3 |.

            A_squared_inv_b = A_squared_inv.dot(b)
            b_A_squared_inv_b = b.T.dot(A_squared_inv_b)

            s_2 = (A_squared_inv + (A_squared_inv_b.dot(A_squared_inv_b.T)) / (a_squared - b_A_squared_inv_b))
            s_2 = - s_2.dot(b) / a_squared

            s_3 = float((1 - b.T.dot(s_2)) / a_squared)

            S_1 = A_squared_inv - A_squared_inv_b.dot(s_2.T)

            B_squared_inv = cp.hstack((S_1, s_2))
            B_squared_inv = cp.vstack((B_squared_inv, cp.vstack((s_2, s_3)).T))

    return B, B_squared, B_squared_inv



