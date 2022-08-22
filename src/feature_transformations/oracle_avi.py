import cupy as cp
import numpy as np
from src.auxiliary_functions.auxiliary_functions import fd
from src.feature_transformations.term_ordering_strategies import pearson
from src.oracles.accelerated_gradient_descent import AcceleratedGradientDescent
from src.oracles.conditional_gradients import ConditionalGradients
from src.oracles.feasible_regions import L1Ball, L2Ball
from src.oracles.objective_functions import L2Loss
from src.feature_transformations.auxiliary_functions_avi import update_coefficient_vectors, streaming_matrix_updates
from src.feature_transformations.terms_and_polynomials import SetsOAndG


class OracleAVI:
    """
    The OAVI algorithm.

    Args:
        psi: float, Optional
            Determines how stronlgy vanishing the polynomials are going to be. (Default is 0.1.)
        eps: float, Optional
            Determines the accuracy of the oracle. (Default is 0.001.)
        tau: float, Optional
            Determines the radius of CCOP. (Default is 1000.)
        lmbda: float, Optional
            The L2 regularization parameter. (Default is 0.0.)
        tol: float, Optional
            If no improvement in loss over the last iteration of at least psi * tol is made, terminates the algorithm.
            (Default is 0.000001.)
        max_degree: int, Optional
            Maximum degree of the polynomials we construct. (Default is 10.)
        objective_type: str, Optional
            (Default is "L2Loss".)
        region_type: str, Optional
            (Default is"L1Ball".)
        oracle_type: str, Optional
            Options are in ["CG", "PCG", "BPCG", "AGD", "ABM"]. Technically, "ABM" does not lead to a new oracle but to
            the ABM algorithm. (Default is "CG".)
        max_iterations: int, Optional
            The maximum number of iterations we run the oracle for, if applicable. (Default is 10000.)
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
        inverse_hessian_boost: str, Optional
            Options include "false", "weak", and "full". Whether to boost the performance of the algorithm using inverse
            hessian information. For the L1Ball, "full" will destroy any sparsity properties, whereas "weak" is a
            compromise between "full" and "false". (Default is "false".)


    Methods:
        determine_term_ordering(data: cp.ndarray)
            Determines an ordering of the data according to term_ordering_strategy.
        fit(X_train: cp.ndarray)
            Creates an OAVI feature transformation fitted to X_train.
        call_oracle(data: cp.ndarray, data_squared: cp.ndarray, data_labels: cp.ndarray, labels: cp.ndarray,
                    labels_squared: float, data_squared_inverse: cp.ndarray)
            Calls the convex oracle.
        evaluate(X_test: cp.ndarray)
            Applies the OAVI feature transformation to X_test.
        evaluate_transformation()
            Evaluates the transformation corresponding to the polynomials in G_poly.

    References:
        [1] Wirth, E. and Pokutta, S., 2022. Conditional Gradients for the Approximately Vanishing Ideal.
        arXiv preprint arXiv:2202.03349.
    """

    def __init__(self,
                 psi: float = 0.1,
                 eps: float = 0.001,
                 tau: float = 1000,
                 lmbda: float = 0.0,
                 tol: float = 0.000001,
                 max_degree: int = 10,
                 objective_type: str = "L2Loss",
                 region_type: str = "L1Ball",
                 oracle_type: str = "CG",
                 max_iterations: int = 10000,
                 term_ordering_strategy: str = "pearson",
                 border_type: str = "gb",
                 inverse_hessian_boost: str = "false"):
        self.psi = psi
        self.eps = self.psi * eps
        self.tau = tau
        self.lmbda = lmbda
        self.tol = tol
        self.max_degree = max_degree
        self.objective_type = objective_type
        self.region_type = region_type
        self.oracle_type = oracle_type
        self.max_iterations = max_iterations

        self.sets_avi = None
        self.degree = 0

        self.term_ordering_strategy = term_ordering_strategy
        self.term_ordering = None

        self.border_type = border_type

        self.inverse_hessian_boost = inverse_hessian_boost
        if self.inverse_hessian_boost is "full":
            assert self.oracle_type in ["CG", "AGD"], ("Inverse Hessian boosting is only available for 'CG' or 'AGD'."
                                                       "Try 'weak' for a speed-up with 'PCG' or 'BPCG'.")
        elif self.inverse_hessian_boost is "weak":
            assert self.oracle_type in ["CG", "PCG", "BPCG"], ("Weak inverse Hessian boosting is only available for"
                                                               "'CG', 'PCG', or 'BPCG'.")

    def determine_term_ordering(self, data: cp.ndarray):
        """Determines an ordering of the data according to term_ordering_strategy."""
        if self.term_ordering_strategy == "deglex":
            self.term_ordering = list(range(data.shape[1]))
        elif self.term_ordering_strategy == "pearson":
            self.term_ordering = pearson(data, rev=False)
        elif self.term_ordering_strategy == "rev pearson":
            self.term_ordering = pearson(data, rev=True)

    def fit(self, X_train: cp.ndarray):
        """Creates an OAVI feature transformation fitted to X_train.

        Args:
            X_train: cp.ndarray
                Training data.

        Returns:
            X_train_transformed: cp.ndarray
            self.sets_avi: instance of SetsOAndG
        """
        # First, the algorithm determines the ordering of the terms to maximize the number of vanishing polynomials.
        self.determine_term_ordering(X_train)

        X_train = X_train[:, self.term_ordering]
        self.degree = 0
        self.sets_avi = SetsOAndG(X_train, border_type=self.border_type)

        while self.degree < self.max_degree:
            self.degree += 1

            border_terms, border_evaluations = self.sets_avi.construct_border()
            O_indices = []
            leading_terms = []
            G_coefficient_vectors = None

            data = fd(self.sets_avi.O_array_evaluations)
            data_squared = data.T.dot(data)
            data_squared_inverse = None

            if self.inverse_hessian_boost in ["weak", "full"]:
                data_squared_inverse = cp.array(np.linalg.inv(data_squared))

            for column_index in range(0, border_terms.shape[1]):
                if G_coefficient_vectors is not None:
                    G_coefficient_vectors = cp.vstack((G_coefficient_vectors,
                                                       cp.zeros((1, G_coefficient_vectors.shape[1]))))

                term_evaluated = fd(border_evaluations[:, column_index])
                data_term_evaluated = data.T.dot(term_evaluated)
                term_evaluated_squared = float(term_evaluated.T.dot(term_evaluated))
                coefficient_vector, loss = self.call_oracle(data, data_squared, data_term_evaluated, term_evaluated,
                                                            term_evaluated_squared, data_squared_inverse)
                if self.inverse_hessian_boost is ['false']:
                    data_squared_inverse = None
                # If polynomial vanishes, append the polynomial to G, otherwise append the leading term to O.
                if loss <= self.psi:
                    leading_terms.append(int(column_index))
                    G_coefficient_vectors = update_coefficient_vectors(G_coefficient_vectors, coefficient_vector)
                else:
                    O_indices.append(int(column_index))
                    data, data_squared, data_squared_inverse = streaming_matrix_updates(
                        data, data_squared, data_term_evaluated, term_evaluated, term_evaluated_squared,
                        data_squared_inverse)

            self.sets_avi.update_leading_terms(fd(border_terms[:, leading_terms]))
            self.sets_avi.update_G(G_coefficient_vectors)

            if not O_indices:
                break
            else:
                self.sets_avi.update_O(fd(border_terms[:, O_indices]), fd(border_evaluations[:, O_indices]), O_indices)
        X_train_transformed = self.sets_avi.G_evaluations
        if X_train_transformed is not None:
            X_train_transformed = cp.abs(X_train_transformed)
        else:
            X_train_transformed = None
        return X_train_transformed, self.sets_avi

    def call_oracle(self, data: cp.ndarray, data_squared: cp.ndarray, data_labels: cp.ndarray, labels: cp.ndarray,
                    labels_squared: float, data_squared_inverse: cp.ndarray = None):
        """Calls the convex oracle.

        Args:
            data: cp.ndarray
            data_squared: cp.ndarray
            data_labels: cp.ndarray
            labels: cp.ndarray
            labels_squared: float
            data_squared_inverse: cp.ndarray, Optional

        Returns:
            coefficient_vector: cp.ndarray
            loss: float
        """
        if self.oracle_type is not "ABM":
            if self.objective_type == "L2Loss":
                objective = L2Loss(data, labels, self.lmbda, data_squared=data_squared,
                                   data_labels=data_labels,
                                   labels_squared=labels_squared, data_squared_inverse=data_squared_inverse)

            if self.region_type == "L1Ball":
                region = L1Ball(data.shape[1], self.tau - 1)
                if self.inverse_hessian_boost is not "false":
                    if cp.linalg.norm(objective.solution, ord=1) > self.tau - 1:
                        print("IHB is stopped. Switching to BPCG.")
                        self.oracle_type = "BPCG"
                        self.inverse_hessian_boost = "false"
            elif self.region_type == "L2Ball":
                region = L2Ball(data.shape[1], self.tau - 1)

            if self.inverse_hessian_boost == "weak":
                assert self.oracle_type in ["CG", "PCG",
                                            "BPCG"], "WIHB is only implemented for Conditional Gradients algorithms."

                oracle = ConditionalGradients(objective_function=objective,
                                              feasible_region=region,
                                              oracle_type="CG",
                                              psi=self.psi,
                                              eps=self.eps,
                                              max_iterations=self.max_iterations,
                                              tol=self.tol,
                                              inverse_hessian_boost="full")
                tmp_coefficient_vector, loss_list, _ = oracle.optimize()
                loss = float(loss_list[-1])
                if loss <= self.psi:
                    oracle = ConditionalGradients(objective_function=objective,
                                                  feasible_region=region,
                                                  oracle_type=self.oracle_type,
                                                  psi=self.psi,
                                                  eps=self.eps,
                                                  max_iterations=self.max_iterations,
                                                  tol=self.tol,
                                                  inverse_hessian_boost="false")
                    tmp_coefficient_vector_2, loss_list_2, _ = oracle.optimize()
                    loss_2 = float(loss_list_2[-1])

                    if loss_2 <= self.psi:
                        tmp_coefficient_vector = tmp_coefficient_vector_2
                        loss = loss_2
                coefficient_vector = fd(cp.vstack((fd(tmp_coefficient_vector), fd(cp.array([[1.0]])))))
            else:
                if self.oracle_type in ["CG", "PCG", "BPCG"]:
                    oracle = ConditionalGradients(objective_function=objective,
                                                  feasible_region=region,
                                                  oracle_type=self.oracle_type,
                                                  psi=self.psi,
                                                  eps=self.eps,
                                                  max_iterations=self.max_iterations,
                                                  tol=self.tol,
                                                  inverse_hessian_boost=self.inverse_hessian_boost)

                elif self.oracle_type == "AGD":
                    oracle = AcceleratedGradientDescent(objective_function=objective,
                                                        psi=self.psi,
                                                        dimension=data.shape[1],
                                                        max_iterations=self.max_iterations,
                                                        tol=self.tol,
                                                        inverse_hessian_boost=self.inverse_hessian_boost)

                tmp_coefficient_vector, loss_list, _ = oracle.optimize()
                del region, objective, oracle
                loss = float(loss_list[-1])
                coefficient_vector = fd(cp.vstack((fd(tmp_coefficient_vector), fd(cp.array([[1.0]])))))
        # ABM
        else:
            data_with_labels = cp.hstack((data, labels))
            if data_with_labels.shape[0] > data_with_labels.shape[1]:
                data_with_labels_squared = cp.hstack((data_squared, data_labels))
                bottom_row = cp.vstack((data_labels, labels_squared)).T
                data_with_labels_squared = cp.vstack((data_with_labels_squared, bottom_row))
                U, S, V = cp.linalg.svd(data_with_labels_squared)
            else:
                U, S, V = cp.linalg.svd(data_with_labels)
            coefficient_vector = fd(V.T[:, -1])
            loss = (1 / data.shape[0]) * cp.linalg.norm(data_with_labels.dot(coefficient_vector)) ** 2
        return coefficient_vector, loss

    def evaluate(self, X_test: cp.ndarray):
        """Applies the OAVI feature transformation to X_test."""
        X_test = X_test[:, self.term_ordering]
        X_test_transformed, test_sets_avi = self.sets_avi.apply_G_transformation(X_test)
        if X_test_transformed is not None:
            X_test_transformed = cp.abs(X_test_transformed)
        else:
            X_test_transformed = None
        return X_test_transformed, test_sets_avi

    def evaluate_transformation(self):
        """Evaluates the transformation corresponding to the polynomials in G_poly."""
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms, degree
         ) = self.sets_avi.evaluate_transformation()
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                degree)
