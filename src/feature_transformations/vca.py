import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.terms_and_polynomials import SetsVCA
from src.feature_transformations.vanishing_polynomial_construction import find_range_null_vca


class VCA:
    """The VCA algorithm.

    Args:
        psi: float, Optional
            Determines how stronlgy vanishing the polynomials are going to be. (Default is 0.1.)
        max_degree: int, Optional
            Maximum degree of the polynomials we construct. (Default is 10.)

    Methods:
        fit(X_train: cp.ndarray)
            Creates a VCA feature transformation fitted to X_train.
        evaluate(X_test: cp.ndarray)
            Applies the VCA feature transformation to X_test.
        evaluate_transformation()
            Evaluates the transformation corresponding to the polynomials in V.

    References:
        [1] Livni, R., Lehavi, D., Schein, S., Nachliely, H., Shalev-Shwartz, S. and Globerson, A., 2013, February.
        Vanishing component analysis. In International Conference on Machine Learning (pp. 597-605). PMLR.
    """

    def __init__(self, psi: float = 0.1, max_degree: int = 10):
        self.psi = psi
        self.max_degree = max_degree

        self.sets_vca = None
        self.sets_vca_test = None
        self.degree = 1

    def fit(self, X_train: cp.ndarray):
        """Creates a VCA feature transformation fitted to X_train.

        Args:
            X_train: cp.ndarray
                Training data.

        Returns:
            X_train_transformed: cp.ndarray
            self.sets_vca: instance of SetsVCA
        """
        self.sets_vca = SetsVCA(X_train)
        self.degree = 1
        while self.degree < self.max_degree:
            border = fd(self.sets_vca.construct_border(degree=self.degree))

            if border is None:
                break
            self.sets_vca.update_C(border)

            V_coefficients, V_evaluations, F_coefficients, F_evaluations = find_range_null_vca(
                cp.hstack(self.sets_vca.Fs), self.sets_vca.Cs[-1], psi=self.psi)
            self.sets_vca.update_V(V_coefficients, V_evaluations)
            if isinstance(F_coefficients, list):
                break
            elif isinstance(F_coefficients, cp.ndarray):
                self.sets_vca.update_F(F_coefficients, F_evaluations)

            self.degree += 1

        X_train_transformed = self.sets_vca.V_to_array()
        if X_train_transformed is not None:
            X_train_transformed = cp.abs(X_train_transformed)
        else:
            X_train_transformed = None
        return X_train_transformed, self.sets_vca

    def evaluate(self, X_test):
        """Applies the VCA feature transformation to X_test."""
        X_test_transformed, self.sets_vca_test = self.sets_vca.apply_V_transformation(X_test)
        if X_test_transformed is not None:
            X_test_transformed = cp.abs(X_test_transformed)
        else:
            X_test_transformed = None
        return cp.abs(X_test_transformed), self.sets_vca_test

    def evaluate_transformation(self):
        """Evaluates the transformation corresponding to the polynomials in V."""
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = self.sets_vca.evaluate_transformation()
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                degree)
