import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.term_ordering_strategies import pearson
from src.feature_transformations.terms_and_polynomials import SetsOAndG
from src.feature_transformations.vanishing_polynomial_construction import find_range_null_avi


class AVI:
    """The AVI algorithm.

    Args:
        psi: float, Optional
            Determines how stronlgy vanishing the polynomials are going to be. (Default is 0.1.)
        tau: float, Optional
            Tolerance controlling sparsity of the feature transformation. (Default is 0.0.)
        max_degree: int, Optional
            Maximum degree of the polynomials we construct. (Default is 10.)
        term_ordering_strategy: str, Optional
            Sort the column vectors of the data set according to this strategy. (Default is "pearson".)
            Options are ["deglex", "pearson", "rev pearson"].
            -   "deglex": standard degree lexicographical ordering.
            -   "pearson": aims to maximize the linear independence of columns in the data, then continues with "deglex"
                at higher degrees.
            -   "rev pearson": aims to minimize the linear independence of columns in the data, then continues with
                "deglex": at higher degrees.
            After the data is sorted according to this method, DegLex is used as the term ordering for higher degrees.
        border_type: str
            "gb": Constructs a Gröbner basis.
            "bb": Constructs a border basis.

    Methods:
        determine_term_ordering(data: cp.ndarray)
            Determines an ordering of the data according to term_ordering_strategy.
        fit(X_train: cp.ndarray)
            Create an AVI feature transformation fitted to X_train.
        evaluate(X_test: cp.ndarray)
            Applies the AVI feature transformation to X_test.
        evaluate_transformation()
            Evaluates the transformation corresponding to the polynomials in G_poly.

    References:
        [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
    """

    def __init__(self, psi: float = 0.1, tau: float = 0.0, max_degree: int = 10,
                 term_ordering_strategy: str = "pearson", border_type: str = "gb"):
        self.psi = psi
        self.tau = tau
        self.max_degree = max_degree
        self.degree = 0

        self.sets_avi = None

        self.term_ordering_strategy = term_ordering_strategy
        self.term_ordering = None

        self.border_type = border_type

    def determine_term_ordering(self, data: cp.ndarray):
        """Determines an ordering of the data according to term_ordering_strategy."""
        if self.term_ordering_strategy == "deglex":
            self.term_ordering = list(range(data.shape[1]))
        elif self.term_ordering_strategy == "pearson":
            self.term_ordering = pearson(data, rev=False)
        elif self.term_ordering_strategy == "rev pearson":
            self.term_ordering = pearson(data, rev=True)

    def fit(self, X_train: cp.ndarray):
        """Create an AVI feature transformation fitted to X_train.

        Args:
            X_train: cp.ndarray
                Training data.

        Returns:
            X_train_transformed: cp.ndarray
            self.avp: instance of ApproximateVanishingPolynomials
            self.nvt: instance of NonVanishingTerms
        """

        # First, the algorithm determines the ordering of the terms to maximize the number of vanishing polynomials.
        self.determine_term_ordering(X_train)

        X_train = X_train[:, self.term_ordering]

        self.sets_avi = SetsOAndG(X_train, border_type=self.border_type)
        self.degree = 0
        while self.degree < self.max_degree:
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
        """Applies the AVI feature transformation to X_test."""
        X_test = X_test[:, self.term_ordering]
        X_test_transformed, test_sets_avi = self.sets_avi.apply_G_transformation(X_test)
        if X_test_transformed is not None:
            X_test_transformed = cp.abs(X_test_transformed)
        else:
            X_test_transformed = None
        return cp.abs(X_test_transformed), test_sets_avi

    def evaluate_transformation(self):
        """Evaluates the transformation corresponding to the polynomials in G_poly."""
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = self.sets_avi.evaluate_transformation()
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                degree)
