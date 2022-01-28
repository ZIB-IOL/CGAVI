import cupy as cp

from src.auxiliary_functions.auxiliary_functions import fd
from src.oracles.accelerated_gradient_descent import AcceleratedGradientDescent
from src.oracles.conditional_gradients import FrankWolfe
from src.oracles.feasibility_regions import L1Ball, L2Ball
from src.oracles.objective_functions import L2Loss
from src.feature_transformations.auxiliary_functions_avi import update_coefficient_vectors
from src.feature_transformations.terms_and_polynomials import SetsAVI


class OracleAVI:
    """
    The oracle based avi algorithm. Constructs generators of a (psi, tau)-approximately vanishing ideal.

    Args:
        psi: float, Optional
            Determines how stronlgy vanishing the polynomials are going to be. (Default is 0.1.)
        eps: float, Optional
            Determines the accuracy of the oracle. (Default is 0.1.)
        tau: float, Optional
            Determines the radius of CCOP. (Default is 10.)
        lmbda: float, Optional
            The L2 regularization parameter. (Default is 0.1.)
        tol: float, Optional
            If no improvement in loss over the last iteration of at least psi * tol is made, terminate the algorithm.
        maximum_degree: int, Optional
            Maximum degree of the polynomials we construct. (Default is 10.)
        objective_type: str, Optional
            (Default is "L2Loss".)
        region_type: str, Optional
            (Default is"L1Ball".)
        oracle_type: str, Optional
            (Default is "FW".)
        max_iterations: int, Optional
            The maximum number of iterations we run the oracle for. (Default is 1000.)

    Methods:
        fit(X_train: cp.ndarray)
            Create an avi feature transformation fitted to data train_or_test X_train.
        call_oracle(data: cp.ndarray, label: cp.ndarray)
            Calls the oracle.
        prepare_evaluation()
            Perform all calculations possible before evaluation.
        evaluate(X_test: cp.ndarray)
            Apply the avi feature transformation to data train_or_test X_test.
        evaluate_sparsity()
            Evaluates the sparsity of the polynomials in G_poly.
    """

    def __init__(self,
                 psi: float = 0.1,
                 eps: float = 0.1,
                 tau: float = 10,
                 lmbda: float = 0.1,
                 tol: float = 0.001,
                 maximum_degree: int = 10,
                 objective_type: str = "L2Loss",
                 region_type: str = "L1Ball",
                 oracle_type: str = "FW",
                 max_iterations: int = 1000):
        self.psi = psi
        self.eps = eps
        self.tau = tau
        self.lmbda = lmbda
        self.tol = tol
        self.maximum_degree = maximum_degree
        self.objective_type = objective_type
        self.region_type = region_type
        self.oracle_type = oracle_type
        self.max_iterations = max_iterations

        self.sets_avi = None
        self.degree = 0

    def fit(self, X_train: cp.ndarray):
        """Create an avi feature transformation fitted to data train_or_test X_train.

        Args:
            X_train: cp.ndarray
                Training data.

        Returns:
            X_train_transformed: cp.ndarray
            self.sets_avi: instance of SetsAVI
        """
        self.degree = 0
        self.sets_avi = SetsAVI(X_train)

        while self.degree < self.maximum_degree:
            self.degree += 1

            border_terms, border_evaluations = self.sets_avi.construct_border()
            O_indices = []
            leading_terms = []
            G_coefficient_vectors = None

            data = fd(self.sets_avi.O_array_evaluations)
            for column_index in range(0, border_terms.shape[1]):
                if G_coefficient_vectors is not None:
                    G_coefficient_vectors = cp.vstack((G_coefficient_vectors,
                                                       cp.zeros((1, G_coefficient_vectors.shape[1]))))
                term_evaluated = fd(border_evaluations[:, column_index])
                tmp = cp.hstack((data, term_evaluated))
                coefficient_vector, loss = self.call_oracle(data, term_evaluated)
                # If polynomial vanishes, append the polynomial to G, otherwise append the leading term to O.
                if loss <= self.psi:
                    leading_terms.append(int(column_index))
                    G_coefficient_vectors = update_coefficient_vectors(G_coefficient_vectors, coefficient_vector)
                else:
                    O_indices.append(int(column_index))
                    data = tmp

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

    def call_oracle(self, data: cp.ndarray, label: cp.ndarray):
        """Calls the oracle.

        Args:
            data: cp.ndarray
            label: cp.ndarray

        Returns:
            coefficient_vector: cp.ndarray
            loss: float
        """
        if self.objective_type == "L2Loss":
            objective = L2Loss(data, label, self.lmbda)

        if self.oracle_type == "FW":
            if self.region_type == "L1Ball":
                region = L1Ball(data.shape[1], self.tau - 1)
            elif self.region_type == "L2Ball":
                region = L2Ball(data.shape[1], self.tau - 1)
            oracle = FrankWolfe(objective_function=objective,
                                feasibility_region=region,
                                psi=self.psi,
                                eps=self.eps,
                                max_iterations=self.max_iterations,
                                tol=self.tol)
            del region
        elif self.oracle_type == "AGD":
            oracle = AcceleratedGradientDescent(objective_function=objective,
                                                psi=self.psi,
                                                dimension=data.shape[1],
                                                max_iterations=self.max_iterations,
                                                tol=self.tol)
        tmp_coefficient_vector, loss_list, _ = oracle.optimize()
        del objective, oracle
        loss = float(loss_list[-1])
        coefficient_vector = fd(cp.vstack((fd(tmp_coefficient_vector), fd(cp.array([[1.0]])))))
        return coefficient_vector, loss

    def evaluate(self, X_test: cp.ndarray):
        """Apply the avi feature transformation to data train_or_test X_test."""
        X_test_transformed, test_sets_avi = self.sets_avi.apply_G_transformation(X_test)
        if X_test_transformed is not None:
            X_test_transformed = cp.abs(X_test_transformed)
        else:
            X_test_transformed = None
        return X_test_transformed, test_sets_avi

    def evaluate_sparsity(self):
        """Evaluates the sparsity of the polynomials in G_poly."""
        (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials
         ) = self.sets_avi.evaluate_sparsity()
        number_of_terms = int(self.sets_avi.O_array_evaluations.shape[1])
        return (total_number_of_zeros, total_number_of_entries, avg_sparsity, number_of_polynomials, number_of_terms,
                self.degree)
