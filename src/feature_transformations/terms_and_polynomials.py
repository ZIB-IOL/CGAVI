import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.auxiliary_functions.indices import find_last_non_zero_entries
from src.auxiliary_functions.sorting import deg_lex_sort
from src.feature_transformations.auxiliary_functions_avi import construct_border


class SetsOAndG:
    """Manages the sets G, O, and borders for OAVI and AVI.

    Args:
        X: cp.ndarray
            The data.
        m: int
        border_type: str
            "gb": Constructs a Gröbner basis.
            "bb": Constructs a border basis.

    Methods:
        update_border(border_terms_raw: cp.ndarray, border_evaluations_raw: cp.ndarray, non_purging_indices: list)
            Updates border.
        update_border_raw(border_terms_raw: cp.ndarray, border_evaluations_raw: cp.ndarray, non_purging_indices: list)
            Updates border_raw.
        update_border_purged(border_terms_purged: cp.ndarray, border_evaluations_purged: cp.ndarray)
            Updates border_purged.
        update_O(O_terms: cp.ndarray, O_evaluations: cp.ndarray, O_indices: list)
            Updates O.
        update_O_terms(O_terms: cp.ndarray)
            Updates O_terms.
        update_O_evaluations(O_evaluations: cp.ndarray)
            Updates O_evaluations.
        update_O_indices(O_indices: list)
            Updates O_indices.
        update_G(G_coefficient_vectors: cp.ndarray)
            Updates G.
        update_leading_terms(leading_terms: cp.ndarray)
            Updates leading terms of generators in G.
        apply_G_transformation(X_test: cp.ndarray)
            Applies the transformation corresponding to G to X_test.
        reconstruct_border(O1_test: cp.ndarray, O_test: cp.ndarray, non_purging_indices: list, X_test: cp.ndarray)
            Reconstructs the border of O_test.
        construct_border()
            Constructs the border of O_terms for the current state of the algorithm.
        evaluate_transformation()
            Evaluates the transformation corresponding to the polynomials in G_coefficient_vectors.

    References:
        [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
        [2] Wirth, E. and Pokutta, S., 2022. Conditional Gradients for the Approximately Vanishing Ideal.
        arXiv preprint arXiv:2202.03349.
    """

    def __init__(self, X: cp.ndarray, m: int = None, border_type: str = "gb"):

        # All lists start at degree 0

        self.X = fd(X)
        if m is not None:
            self.m = m
        else:
            self.m = self.X.shape[0]

        # borders
        self.border_type = border_type
        # List of border terms before purging by degree
        self.border_terms_raw = [None]
        # List of evaluations of borders_terms_raw over X by degree
        self.border_evaluations_raw = [None]
        # List of border terms after purging by degree
        self.border_terms_purged = [None]
        # List of evaluations of border_terms_purged over X by degree
        self.border_evaluations_purged = [None]
        # List of indices such that border_terms_raw[i][:, non_purging_indices[i]] = border_terms_purged[i]
        self.non_purging_indices = [None]

        # O
        # List of O_terms by degree
        self.O_terms = [fd(cp.zeros((self.X.shape[1], 1)))]
        # List of evaluations of O_terms over X by degree
        self.O_evaluations = [fd(cp.ones((self.X.shape[0], 1)))]
        # List of indices in the border that get appended to O
        self.O_indices = [None]
        # Array of all O_terms
        self.O_array_terms = fd(self.O_terms[0])
        # Array of all O_evaluations
        self.O_array_evaluations = fd(self.O_evaluations[0])

        # List of coefficient vectors by degree
        self.G_coefficient_vectors = [None]
        # List of all evaluation vectors of generators of G by degree
        self.G_evaluations = None

        # Example:
        # self.O_terms = [O0, O1, O2, ...], self.border_terms_purged = [None, border1, border2, ...], and
        # self.G_coefficient_vectors = [None, G_coefficient_vectors1, G_coefficient_vectors2, ...]. Then, evaluations
        # of generators of degree 1 are computed via
        # cp.array([O0, O1, border1]) G_coefficient_vectors1.T and valuations of generators of degree 2
        # are cp.array([O0, O1, O2, border2]) cp.array([G_coefficient_vectors1, G_coefficient_vectors2]).T.

        # cp.ndarray of leading terms of polynomials in G
        self.leading_terms = None

    def update_border(self, border_terms_raw: cp.ndarray, border_evaluations_raw: cp.ndarray,
                      non_purging_indices: list):
        """Updates border."""
        self.update_border_raw(fd(border_terms_raw), fd(border_evaluations_raw), non_purging_indices)
        self.update_border_purged(fd(border_terms_raw[:, non_purging_indices]),
                                  fd(border_evaluations_raw[:, non_purging_indices]))

    def update_border_raw(self, border_terms_raw: cp.ndarray, border_evaluations_raw: cp.ndarray,
                          non_purging_indices: list):
        """Updates border_raw."""
        self.border_terms_raw.append(border_terms_raw)
        self.border_evaluations_raw.append(border_evaluations_raw)
        self.non_purging_indices.append(non_purging_indices)

    def update_border_purged(self, border_terms_purged: cp.ndarray, border_evaluations_purged: cp.ndarray):
        """Updates border_purged."""
        self.border_terms_purged.append(border_terms_purged)
        self.border_evaluations_purged.append(border_evaluations_purged)

    def update_O(self, O_terms: cp.ndarray, O_evaluations: cp.ndarray, O_indices: list):
        """Updates O."""
        self.update_O_terms(O_terms)
        self.update_O_evaluations(O_evaluations)
        self.update_O_indices(O_indices)

    def update_O_terms(self, O_terms: cp.ndarray):
        """Updates O_terms."""
        self.O_terms.append(fd(O_terms))
        self.O_array_terms = cp.hstack((fd(self.O_array_terms), fd(O_terms)))

    def update_O_evaluations(self, O_evaluations: cp.ndarray):
        """Updates O_evaluations."""
        self.O_evaluations.append(fd(O_evaluations))
        self.O_array_evaluations = cp.hstack((fd(self.O_array_evaluations), fd(O_evaluations)))

    def update_O_indices(self, O_indices: list):
        """Updates O_indices."""
        self.O_indices.append(O_indices)

    def update_G(self, G_coefficient_vectors: cp.ndarray):
        """Updates G."""
        if G_coefficient_vectors is not None:
            if self.G_evaluations is None:
                self.G_evaluations = fd(cp.hstack((self.O_array_evaluations,
                                                   self.border_evaluations_purged[-1])).dot(G_coefficient_vectors))
            else:
                current_G_evaluations = fd(cp.hstack((self.O_array_evaluations,
                                                      self.border_evaluations_purged[-1])).dot(
                    G_coefficient_vectors))
                self.G_evaluations = cp.hstack((self.G_evaluations, current_G_evaluations))
        self.G_coefficient_vectors.append(G_coefficient_vectors)

    def update_leading_terms(self, leading_terms: cp.ndarray):
        """Updates leading terms of generators in G."""
        if self.leading_terms is None:
            if leading_terms is not None:
                self.leading_terms = fd(leading_terms)
        else:
            self.leading_terms = cp.hstack((self.leading_terms, leading_terms))

    def apply_G_transformation(self, X_test: cp.ndarray):
        """Applies the transformation corresponding to G to X_test."""
        test_sets_avi = SetsOAndG(X_test, m=self.m)
        test_sets_avi.border_evaluations_raw.append(fd(X_test))

        # Degree-1 border
        i = 1
        test_sets_avi.update_border_purged(None, X_test)
        test_sets_avi.update_G(self.G_coefficient_vectors[i])
        if i < len(self.O_evaluations):
            test_sets_avi.update_O_evaluations(fd(X_test[:, self.O_indices[i]]))

        # Higher-degree borders
        i = 2
        while i < min(len(self.O_evaluations) + 1, len(self.non_purging_indices)):
            if self.border_type == "gb":
                border_test_purged = test_sets_avi.reconstruct_border(fd(test_sets_avi.O_evaluations[1]),
                                                                      fd(test_sets_avi.O_evaluations[-1]),
                                                                      non_purging_indices=self.non_purging_indices[i])
            else:
                border_test_purged = test_sets_avi.reconstruct_border(fd(test_sets_avi.border_evaluations_raw[1]),
                                                                      fd(test_sets_avi.O_evaluations[-1]),
                                                                      non_purging_indices=self.non_purging_indices[i])
            test_sets_avi.update_border_purged(None, border_test_purged)

            test_sets_avi.update_G(self.G_coefficient_vectors[i])
            if i < len(self.O_evaluations):
                test_sets_avi.update_O_evaluations(fd(border_test_purged[:, self.O_indices[i]]))
            i += 1

        return test_sets_avi.G_evaluations, test_sets_avi

    def reconstruct_border(self, O1_test: cp.ndarray, O_test: cp.ndarray, non_purging_indices: list,
                           X_test: cp.ndarray = None):
        """Reconstructs the border of O_test."""
        if X_test is None:
            O1_test = fd(O1_test)
            O_test = fd(O_test)
            O1_test_tile = cp.tile(O1_test, (1, O_test.shape[1]))[:, non_purging_indices]
            O_test_repeat = cp.repeat(O_test, repeats=O1_test.shape[1], axis=1)[:, non_purging_indices]
            border_test = cp.multiply(O1_test_tile, O_test_repeat)
        else:
            border_test = X_test
        return fd(border_test)

    def construct_border(self):
        """Constructs the border of O_terms for the current state of the algorithm."""
        if len(self.O_terms) == 1:
            border_terms_raw, border_evaluations_raw, _ = deg_lex_sort(cp.identity(self.X.shape[1]), self.X)
            border_terms_purged, border_evaluations_purged = border_terms_raw, border_evaluations_raw
            non_purging_indices = None
            self.update_border_raw(border_terms_raw, border_evaluations_raw, non_purging_indices)
            self.update_border_purged(border_terms_purged, border_evaluations_purged)
        else:
            if self.border_type == "gb":
                border_terms_raw, border_evaluations_raw, non_purging_indices = construct_border(
                    fd(self.O_terms[-1]), fd(self.O_evaluations[-1]), fd(self.X), fd(self.O_terms[1]),
                    fd(self.O_evaluations[1]), self.leading_terms)
            else:
                border_terms_raw, border_evaluations_raw, non_purging_indices = construct_border(
                    fd(self.O_terms[-1]), fd(self.O_evaluations[-1]), fd(self.X), fd(self.border_terms_raw[1]),
                    fd(self.border_evaluations_raw[1]), None)
            self.update_border(border_terms_raw, border_evaluations_raw, non_purging_indices)

        return self.border_terms_purged[-1], self.border_evaluations_purged[-1]

    def evaluate_transformation(self):
        """Evaluates the transformation corresponding to the polynomials in G_coefficient_vectors.

        Returns:
            total_number_of_zeros: int
                Sum of all zero entries in coefficient vectors in G_coefficient_vectors.
            total_number_of_entries: int
                Total number of entries in coefficient vectors in G_coefficient_vectors.
            avg_sparsity: float
                The average sparsity of coefficient vectors in G_coefficient_vectors.
            number_of_polynomials: int
                Number of polynomials in G.
            number_of_terms: int
                Number of terms in O.
            degree: int
                Average degree of polynomials in G.
        """
        number_of_polynomials, total_number_of_zeros, total_number_of_entries, avg_sparsity, degree = 0, 0, 0, 0.0, 0
        for i in range(0, len(self.G_coefficient_vectors)):
            coefficient_vectors = self.G_coefficient_vectors[i]
            if coefficient_vectors is not None:
                if not isinstance(coefficient_vectors, cp.ndarray):
                    coefficient_vectors = coefficient_vectors.toarray()
                coefficient_vectors = cp.array(coefficient_vectors)
                degree += i * coefficient_vectors.shape[1]
                indices = find_last_non_zero_entries(coefficient_vectors)
                for j in range(0, coefficient_vectors.shape[1]):
                    poly = fd(coefficient_vectors[:indices[j] + 1, j])
                    number_of_entries = poly.shape[0] - j
                    number_non_zeros = int(cp.count_nonzero(poly))
                    number_of_zeros = max(number_of_entries - number_non_zeros, 0)
                    sparsity = number_of_zeros / (max(number_of_entries - 1, 1))
                    total_number_of_zeros += number_of_zeros
                    total_number_of_entries += number_of_entries
                    avg_sparsity += sparsity
                    number_of_polynomials += 1
        if number_of_polynomials != 0:
            avg_sparsity = avg_sparsity / number_of_polynomials
        degree = degree / number_of_polynomials
        number_of_terms = self.O_array_evaluations.shape[1]
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                degree)


class SetsVCA:
    """
    Manages the sets V, C, and F for VCA.

    Args:
        X: cp.ndarray
            The data.
        m: int, Optional
            Number of data points. If not provided, this is computed from X. (Default is None.)


    Methods:
        update_F(F_coefficient_vectors: cp.ndarray, F_evaluation_vectors: cp.ndarray)
            Updates F.
        F_to_array()
            Transforms F into one array.
        update_V(V_coefficient_vectors: cp.ndarray, V_evaluation_vectors: cp.ndarray)
            Updates V.
        V_to_array()
            Transforms V into one array.
        apply_V_transformation(X_test: cp.ndarray)
            Applies the transformation corresponding to the V polynomials to X_test.
        update_C(vectors: cp.ndarray)
            Updates C.
        construct_border(degree: int)
            Constructs the border for the current state of the algorithm.
        evaluate_transformation()
            Evaluates the transformation corresponding to the polynomials in V w.r.t. to the functions in F and C.

    References:
        [1] Livni, R., Lehavi, D., Schein, S., Nachliely, H., Shalev-Shwartz, S. and Globerson, A., 2013, February.
        Vanishing component analysis. In International Conference on Machine Learning (pp. 597-605). PMLR.
    """
    def __init__(self, X: cp.ndarray, m: int = None):

        # C
        self.X = fd(X)
        self.Cs = []

        # F
        if m is None:
            self.m = self.X.shape[0]
        else:
            self.m = m
        self.Fs = [fd(cp.ones((self.X.shape[0], 1)) / cp.sqrt(self.m))]
        self.F_coefficient_vectors = [cp.array([[1]])]

        # V
        self.Vs = []
        self.V_coefficient_vectors = []

    def update_F(self, F_coefficient_vectors: cp.ndarray, F_evaluation_vectors: cp.ndarray):
        """Updates F."""
        assert (isinstance(F_coefficient_vectors, cp.ndarray), "New Fs should not be empty.")
        assert isinstance(F_evaluation_vectors, cp.ndarray), "New Fs should not be empty."
        self.F_coefficient_vectors.append(fd(F_coefficient_vectors))
        self.Fs.append(fd(F_evaluation_vectors))

    def F_to_array(self):
        """Transforms F into one array."""
        F_array = None
        for F in self.Fs:
            if isinstance(F, cp.ndarray):
                if F_array is None:
                    F_array = fd(F)
                else:
                    F_array = cp.hstack((F_array, fd(F)))
        return F_array

    def update_V(self, V_coefficient_vectors: cp.ndarray, V_evaluation_vectors: cp.ndarray):
        """Updates V."""
        V_vectors = None
        V_evaluations = None

        if isinstance(V_coefficient_vectors, cp.ndarray):
            V_vectors = fd(V_coefficient_vectors)
            assert (isinstance(V_evaluation_vectors, cp.ndarray), "Evaluation vectors have to be cp.arrays.")
            V_evaluations = fd(V_evaluation_vectors)
        self.Vs.append(V_evaluations)
        self.V_coefficient_vectors.append(V_vectors)

    def V_to_array(self):
        """Transforms V into one array."""
        V_array = None
        for V in self.Vs:
            if isinstance(V, cp.ndarray):
                if V_array is None:
                    V_array = fd(V)
                else:
                    V_array = cp.hstack((V_array, fd(V)))
        return V_array

    def apply_V_transformation(self, X_test: cp.ndarray):
        """Applies the transformation corresponding to the V polynomials to X_test."""
        sets_VCA_test = SetsVCA(X_test, self.X.shape[0])
        degree = 1
        while degree < len(self.Cs) + 1:
            border = fd(sets_VCA_test.construct_border(degree=degree))
            sets_VCA_test.update_C(border)
            degree += 1

            V_coefficients_test = None
            if degree - 2 < len(self.V_coefficient_vectors):
                V_coefficients_test = self.V_coefficient_vectors[degree - 2]

            if isinstance(V_coefficients_test, cp.ndarray):
                V_evalutations_test = (fd(V_coefficients_test).T.dot(
                    (cp.hstack((fd(cp.hstack(sets_VCA_test.Fs)), fd(sets_VCA_test.Cs[-1])))).T)).T
            else:
                V_evalutations_test = None
            sets_VCA_test.update_V(V_coefficients_test, V_evalutations_test)

            F_coefficients_test = None
            if degree - 1 < len(self.F_coefficient_vectors):
                F_coefficients_test = fd(self.F_coefficient_vectors[degree - 1])

            if isinstance(F_coefficients_test, cp.ndarray):
                F_evaluations_test = (fd(F_coefficients_test).T.dot(
                    (cp.hstack((fd(cp.hstack(sets_VCA_test.Fs)), fd(sets_VCA_test.Cs[-1])))).T)).T
                sets_VCA_test.update_F(F_coefficients_test, F_evaluations_test)
            else:
                break
        X_test_transformed = sets_VCA_test.V_to_array()
        return X_test_transformed, sets_VCA_test

    def update_C(self, vectors: cp.ndarray):
        """Updates C."""
        assert isinstance(vectors, cp.ndarray), "Requires cp.ndarray."
        self.Cs.append(fd(vectors))

    def construct_border(self, degree: int = 1):
        """Constructs the border for the current state of the algorithm."""
        if degree != 1:
            F1 = self.Fs[1]
            F_current = self.Fs[-1]
            F1_tile = cp.tile(F1, (1, F_current.shape[1]))
            F_current_repeat = cp.repeat(F_current, repeats=F1.shape[1], axis=1)
            border = cp.multiply(F1_tile, F_current_repeat)
        else:
            border = self.X
        return border

    def evaluate_transformation(self):
        """Evaluates the transformation corresponding to the polynomials in V w.r.t. to the functions in F and C.

        Returns:
            total_number_of_zeros: int
                Sum of all zero entries in coefficient vectors in V_coefficient_vectors.
            total_number_of_entries: int
                Total number of entries in coefficient vectors in V_coefficient_vectors.
            avg_sparsity: float
                The average sparsity of coefficient vectors in V_coefficient_vectors.
            number_of_polynomials: int
                Number of polynomials in V_coefficient_vectors.
            degree: int
                Average degree of polynomials in V.
        """
        total_number_of_zeros, total_number_of_entries, avg_sparsity, degree = 0, 0, 0.0, 0
        number_of_polynomials = 0

        for i in range(0, len(self.V_coefficient_vectors)):
            coefficient_vectors = self.V_coefficient_vectors[i]
            if coefficient_vectors is not None:
                if not isinstance(coefficient_vectors, cp.ndarray):
                    coefficient_vectors = coefficient_vectors.toarray()
                coefficient_vectors = cp.array(coefficient_vectors)
                degree += (i+1) * coefficient_vectors.shape[1]

                coefficient_vectors = fd(coefficient_vectors)
                for j in range(0, coefficient_vectors.shape[1]):
                    number_of_polynomials += 1
                    poly = fd(coefficient_vectors[:, j])
                    number_of_entries = len(poly)
                    number_of_zeros = number_of_entries - int(cp.count_nonzero(poly))
                    sparsity = number_of_zeros / (max(number_of_entries - 1, 1))
                    total_number_of_zeros += number_of_zeros
                    total_number_of_entries += number_of_entries
                    avg_sparsity += sparsity
        avg_sparsity = avg_sparsity / number_of_polynomials
        degree = degree / number_of_polynomials
        number_of_terms = self.F_to_array().shape[1]
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                degree)
