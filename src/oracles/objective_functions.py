
import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd


class L2Loss:
    """Represents the l2 loss function f(x) = 1/m||Ax + b||^2 + 1/2 lmbda ||x||^2.

    Attributes:
        data: cp.ndarray
            A cp.ndarray of dimension (m, n).
        labels: cp.ndarray
            A cp.ndarray of dimension (m, ).
        lmbda: float, Optional
            Regularization parameter. (Default is 0.0.)
        data_squared: cp.ndarray, Optional
            A cp.ndarray of dimension (n, n). (Default is None.)
        labels_squared: float, Optional
            (Default is None.)
        data_squared_inverse: cp.ndarray, Optional
            A cp.ndarray of dimension (n, n). (Default is None.)
        data_labels: cp.ndarray, Optional
            A cp.ndarray of dimension (n, ). (Default is None.)

    Methods:
        evaluate_function(x: cp.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: cp.ndarray)
            Evaluates the gradient of f at x.
        evaluate_hessian_inverse()
            Evaluates the inverse of the Hessian of f at x.
        evaluate_step_size(x: cp.ndarray, grad: cp.ndarray, direction: cp.ndarray, step_size_rule: str,
                           iterations_line_search: int, max_step_size: float)
            Computes the step size for iterate x in a certain direction.
        L()
            Computes the smoothness parameter of the loss function.
        alpha()
            Computes the strong convexity parameter of the loss function.
    """
    def __init__(self, data: cp.ndarray, labels: cp.ndarray, lmbda: float = 0.0,
                 data_squared: cp.ndarray = None, labels_squared: float = None,
                 data_squared_inverse: cp.ndarray = None, data_labels: cp.ndarray = None):

        self.A = fd(data)
        self.m, self.n = self.A.shape
        self.b = labels.flatten()
        self.lmbda = float(lmbda)

        # Precomputations of certain matrices and vectors

        # A_squared
        if data_squared is not None:
            self.A_squared = (2 / self.m) * data_squared
        else:
            self.A_squared = (2 / self.m) * self.A.T.dot(self.A)
        if lmbda != 0:
            self.A_squared = self.A_squared + lmbda * cp.identity(self.n)

        # A_squared_inv
        self.A_squared_inv = None
        self.solution = None
        if data_squared_inverse is not None:
            self.A_squared_inv = (self.m / 2) * data_squared_inverse
            self.solution = - data_squared_inverse.dot(data_labels).flatten()
            assert lmbda == 0, "Regularization is not implemented for hessian-based algorithms"

        # A.Tb
        if data_labels is not None:
            self.A_b = (2 / self.m) * data_labels
        else:
            self.A_b = (2 / self.m) * self.A.T.dot(self.b[:, cp.newaxis])

        # b.T.b
        if labels_squared is not None:
            self.b_squared = (2 / self.m) * labels_squared
        else:
            self.b_squared = float((2 / self.m) * self.b.T.dot(self.b))

    def evaluate_function(self, x: cp.ndarray):
        """Evaluates f at x."""
        return float((1 / 2) * x.T.dot(self.A_squared).dot(x) + self.A_b.T.dot(x) + (1 / 2) * self.b_squared)

    def evaluate_gradient(self, x: cp.ndarray):
        """Evaluates the gradient of f at x."""
        return self.A_squared.dot(fd(x)) + self.A_b

    def evaluate_hessian_inverse(self):
        """Evaluates the inverse of the Hessian of f at x."""
        return self.A_squared_inv

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
                Point in the feasible region.
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
                                    / (direction.T.dot(self.A_squared).dot(direction)), max_step_size)

        elif step_size_rule == "line_search":
            assert iterations_line_search > 0, "Number of iterations needs to be greater than 0."
            optimal_value = 1e16
            for i in range(0, iterations_line_search):
                current_step_size = float(i / iterations_line_search)
                tmp = x + current_step_size * direction
                current_value = self.evaluate_function(tmp)
                if current_value < optimal_value:
                    optimal_value = current_value
                    optimal_step_size = current_step_size

        return optimal_step_size

    def L(self):
        """Computes the smoothness parameter of the loss function."""
        eigenvalues, _ = cp.linalg.eigh(self.A_squared)
        return cp.max(eigenvalues)

    def alpha(self):
        """Computes the strong convexity parameter of the loss function."""
        eigenvalues, _ = cp.linalg.eigh(self.A_squared)
        return cp.min(eigenvalues)
