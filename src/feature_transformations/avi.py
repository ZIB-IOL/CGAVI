import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.terms_and_polynomials import SetsAVI
from src.feature_transformations.vanishing_polynomial_construction import find_range_null_avi


class AVI:
    """
    The avi algorithm.

    Args:
        psi: float, Optional
            Determines how stronlgy vanishing the polynomials are going to be. (Default is 0.1.)
        tau: float, Optional
            Tolerance controlling sparsity of the feature transformation. (Default is 0.1.)
        maximum_degree: int, Optional
            Maximum degree of the polynomials we construct. (Default is 10.)

    Methods:
        fit(X_train: cp.ndarray)
            Create an avi feature transformation fitted to data train_or_test X_train.
        prepare_evaluation()
            Perform all calculations possible before evaluation.
        evaluate(X_test: cp.ndarray)
            Apply the avi feature transformation to data train_or_test X_test.
        evaluate_sparsity()
            Evaluates the sparsity of the polynomials in G_poly.

    References:
        Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
    """

    def __init__(self, psi: float = 0.1, tau: float = 10, maximum_degree: int = 10):
        self.psi = psi
        self.tau = tau
        self.maximum_degree = maximum_degree
        self.degree = 0
        self.sets_avi = None

    def fit(self, X_train: cp.ndarray):
        """Create an avi feature transformation fitted to data X_train.

        Args:
            X_train: cp.ndarray
                Training data.

        Returns:
            X_train_transformed: cp.ndarray
            self.avp: instance of ApproximatelyVanishingPolynomials
            self.nvt: instance of NonVanishingTerms
        """
        self.sets_avi = SetsAVI(X_train)
        self.degree = 0
        while self.degree < self.maximum_degree:
            self.degree += 1

            O_terms = fd(self.sets_avi.O_array_terms)
            O_evaluations = fd(self.sets_avi.O_array_evaluations)

            border_terms, border_evaluations = self.sets_avi.construct_border()
            (G_coefficient_vectors, new_leading_terms, new_O_terms, new_O_evaluations, new_O_indices
             ) = find_range_null_avi(O_terms, O_evaluations, border_terms, border_evaluations,
                                     psi=self.psi, tau=self.tau)

            self.sets_avi.update_leading_terms(new_leading_terms)
            self.sets_avi.update_G(G_coefficient_vectors)
            if not new_O_indices:
                break

            else:
                self.sets_avi.update_O(fd(new_O_terms), fd(new_O_evaluations), new_O_indices)

        X_train_transformed = self.sets_avi.G_evaluations
        if X_train_transformed is not None:
            X_train_transformed = cp.abs(X_train_transformed)
        else:
            X_train_transformed = None
        return cp.abs(X_train_transformed), self.sets_avi

    def evaluate(self, X_test: cp.ndarray):
        """Apply the avi feature transformation to data train_or_test X_test."""
        X_test_transformed, test_sets_avi = self.sets_avi.apply_G_transformation(X_test)
        if X_test_transformed is not None:
            X_test_transformed = cp.abs(X_test_transformed)
        else:
            X_test_transformed = None
        return cp.abs(X_test_transformed), test_sets_avi

    def evaluate_sparsity(self):
        """Evaluates the sparsity of the polynomials in G_poly."""
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials
         ) = self.sets_avi.evaluate_sparsity()
        number_of_terms = int(self.sets_avi.O_array_evaluations.shape[1])
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                self.degree)
