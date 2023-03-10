import os
import sys
import cupy as cp
import numpy as np


def fd(x):
    """Guarantees that x is 2-dimensional."""
    array_type = type(x)
    if x.ndim == 1:
        if array_type is np.ndarray:
            x = x.flatten()[:, np.newaxis]
        elif array_type is cp.ndarray:
            x = x.flatten()[:, cp.newaxis]
    return x


def orthogonal_projection(vectors: cp.ndarray, vector: cp.ndarray):
    """Obtains orthogonal components of vector projected onto vectors.

    Args:
        vectors: cp.ndarray
        vector: cp.ndarray

    Returns:
        orthogonal_components: cp.ndarray
            A column vector of the form (vector.T.dot(vectors)).T.
    """
    vector = fd(vector)
    vectors = fd(vectors)
    assert vectors.shape[0] == vector.shape[0], "Vectors have to be of identical length."
    orthogonal_components = (vector.T.dot(vectors)).T
    return fd(orthogonal_components)


def translate_names(hyperparameters, ordering: bool = True, inverse_hessian_boost: bool = True,
                    border_type: bool = True, tex: bool = False):
    """Given hyperparameters, creates the correct name of the algorithm."""
    if tex:
        if hyperparameters["algorithm"] == "avi":
            algorithm_name = r'$\texttt{AVI}$'
        elif hyperparameters["algorithm"] == "vca":
            algorithm_name = r'$\texttt{VCA}$'
        elif hyperparameters["algorithm"] == "svm":
            algorithm_name = r'$\texttt{SVM}$'
        elif hyperparameters["algorithm"] == "oavi":
            if hyperparameters["oracle_type"] == "ABM":
                algorithm_name = r'$\texttt{ABM}$'
            else:
                algorithm_name = hyperparameters["oracle_type"] + "AVI"
                algorithm_name = r'$\texttt{{{replace}}}$'.format(replace=algorithm_name)
                # Inverse Hessian Boosting
                if inverse_hessian_boost:
                    if hyperparameters["inverse_hessian_boost"] == "full":
                        algorithm_name = algorithm_name + "-" + r'$\texttt{IHB}$'
                    elif hyperparameters["inverse_hessian_boost"] == "weak":
                        algorithm_name = algorithm_name + "-" + r'$\texttt{WIHB}$'
                    elif hyperparameters["inverse_hessian_boost"] == "false":
                        pass
        # border type
        if hyperparameters["algorithm"] in ['oavi', 'avi'] and border_type:
            if hyperparameters["border_type"] == "gb":
                algorithm_name = algorithm_name + "-" + r'$\texttt{GB}$'
            else:
                algorithm_name = algorithm_name + "-" + r'$\texttt{BB}$'

    if not tex:
        if hyperparameters["algorithm"] == "avi":
            algorithm_name = 'AVI'
        elif hyperparameters["algorithm"] == "vca":
            algorithm_name = 'VCA'
        elif hyperparameters["algorithm"] == "svm":
            algorithm_name = 'SVM'
        elif hyperparameters["algorithm"] == "oavi":
            if hyperparameters["oracle_type"] == "ABM":
                algorithm_name = 'ABM'
            else:
                algorithm_name = hyperparameters["oracle_type"] + "AVI"
                # hessian boosting
                if inverse_hessian_boost:
                    if hyperparameters["inverse_hessian_boost"] == "full":
                        algorithm_name = algorithm_name + "-" + 'IHB'
                    elif hyperparameters["inverse_hessian_boost"] == "weak":
                        algorithm_name = algorithm_name + "-" + 'WIHB'
                    elif hyperparameters["inverse_hessian_boost"] == "false":
                        pass

        # border type
        if hyperparameters["algorithm"] in ['oavi', 'avi'] and border_type:
            if hyperparameters["border_type"] == "gb":
                algorithm_name = algorithm_name + "-" + "gb"
            else:
                algorithm_name = algorithm_name + "-" + "bb"

    # term ordering strategy
    if hyperparameters["algorithm"] == "oavi" and ordering:
        if hyperparameters["term_ordering_strategy"] == "pearson":
            algorithm_name = algorithm_name + "-" + "p"
        elif hyperparameters["term_ordering_strategy"] == "rev_pearson":
            algorithm_name = algorithm_name + "-" + "pr"
        elif hyperparameters["term_ordering_strategy"] == "deglex":
            pass

    return algorithm_name


def G_O_bound(psi, n):
    """Computes a bound on |G| + |O| <= binom(D + n)(D), where D = lceil -log(psi) / log(4) rceil"""
    D = np.ceil(-np.log(psi) /np.log(4))
    return np.math.factorial(D + n) /(np.math.factorial(D) * np.math.factorial(n))


G_O_bound_vectorized = np.vectorize(G_O_bound)


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
