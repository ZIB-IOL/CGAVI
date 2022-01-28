import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd


class L2Loss:
    """Represents the l2 loss function f(x) = 1/m||Ax + b||^2 + 1/2 lmbda ||x||^2.

    Attributes:
        data_matrix: cp.ndarray
            A cp.ndarray of dimension (m, n).
        labels: cp.ndarray
            A cp.ndarray of dimension (m, ).
        lmbda: float, Optional
            Regularization parameter. (Default is 0.0.)

    Methods:
        evaluate_loss(x: cp.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: cp.ndarray)
            Evaluates the gradient of f at x.
        evaluate_step_size(iteration: int, x: cp.ndarray, direction: cp.ndarray, step: dict, max_step: float = 1)
            Computes the step size for iterate x in a certain direction by solving
                argmin_gamma f(x + gamma * direction)
        L()
            Computes the smoothness parameter of the loss function.

    """

    def __init__(self, data_matrix: cp.ndarray, labels: cp.ndarray, lmbda: float = 0.0):

        self.A = fd(data_matrix)
        self.m, self.n = self.A.shape
        self.b = labels.flatten()
        self.lmbda = float(lmbda)

        # Precomputations of certain matrices and vectors
        # Store 2 / m * A^T A +  lmbda * identity
        self.Asquared = 2 / self.m * self.A.T.dot(self.A) + lmbda * cp.identity(self.n)
        # Store 2 / m * A^T b
        self.Ab = 2 / self.m * self.A.T.dot(self.b[:, cp.newaxis])

    def evaluate_loss(self, x: cp.ndarray):
        """Evaluates the loss of f at x."""
        return float((self.lmbda / 2 * cp.linalg.norm(x) ** 2)
                     + (cp.linalg.norm(self.A.dot(fd(x)).flatten() + self.b) ** 2) / self.m)

    def evaluate_gradient(self, x: cp.ndarray):
        """Evaluates the gradient of f at x."""
        return self.Asquared.dot(fd(x)) + self.Ab

    def evaluate_step_size(self,
                           x: cp.ndarray,
                           grad: cp.ndarray,
                           direction: cp.ndarray,
                           step_size_rule: str = "exact",
                           iterations_line_search: int = 1000,
                           max_step_size: float = 1.):
        """Computes the step size for iterate x in a certain direction.

        Args:
            x: cp.ndarray
                Point in the feasibility_region.
            grad: cp.ndarray
                Gradient of f at x.
            direction: cp.ndarray
                vertex - iterate. Direction of the next step.
            step_size_rule: str, Optional
                Either "exact" or "line_search". (Default is "exact".)
            iterations_line_search: int, Optional
                Number of iterations we perform line search for. (Default is 1000.)
            max_step_size: float, Optional
                Maximum step size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step size computed according to the chosen method.
        """
        optimal_step_size = 0

        if step_size_rule == "exact":
            optimal_step_size = min(float(-grad.T.dot(direction))
                                    / (direction.T.dot(self.Asquared).dot(direction)), max_step_size)

        elif step_size_rule == "line_search":
            assert iterations_line_search > 0, "Number of iterations needs to be greater than 0."
            optimal_value = 1e16
            for i in range(0, iterations_line_search):
                current_step_size = float(i / iterations_line_search)
                tmp = x + current_step_size * direction
                current_value = self.evaluate_loss(tmp)
                if current_value < optimal_value:
                    optimal_value = current_value
                    optimal_step_size = current_step_size

        return optimal_step_size

    def L(self):
        """Computes the smoothness parameter of the loss function."""
        eigenvalues, _ = cp.linalg.eigh(self.Asquared)
        return cp.max(eigenvalues)

