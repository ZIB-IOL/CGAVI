import cupy as cp


class AcceleratedGradientDescent:
    """
    The Accelerated Gradient Descent Algorithm

    The algorithm will terminate if finding a psi approximately vanishing polynomial is impossible or as soon as one
    is found.

    Args:
        objective_function: instance of L2Loss
        psi: float, Optional
            (Default is 0.1.)
        n_iter_no_change: int, Optional
            (Default is 5.)
        tol: float, Optional
            If no improvement in loss over the last n_iter_no_change iterations of at least psi * tol is made, terminate
            the algorithm. (Default is 0.05.)
        max_iterations: int, Optional
            Maximum number of iterations. (Default is 100.)
        compute_loss: bool, Optional
            If true, computes loss and uses it as termination criterion. (Default is True.)

    Methods:
        optimize()
            Performs accelerated gradient descent.

    References:
        Nesterov, Y (1983). "A method for unconstrained convex minimization problem with the rate of convergence
        O ( 1 / k 2 ) {\displaystyle O(1/k^{2})} O(1/k^2)". Doklady AN USSR. 269: 543–547.
    """

    def __init__(self,
                 objective_function,
                 psi: float = 0.1,
                 n_iter_no_change: int = 3,
                 tol: float = 0.001,
                 dimension: int = 10,
                 max_iterations: int = 10000,
                 compute_loss: bool = True):
        self.objective = objective_function
        self.L = objective_function.L()
        self.psi = psi
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.dimension = dimension
        self.max_iterations = max_iterations
        self.compute_loss = compute_loss

    def optimize(self):
        """Performs accelerated gradient descent.
        Returns:
            y: cp.ndarray
            loss_list: list
            None: dummy output
        """
        x = cp.zeros((self.dimension, 1)).flatten()
        y = x.copy()
        z = x.copy()
        loss_list = []
        iter_no_change = 0
        for epoch in range(0, self.max_iterations):
            grad = self.objective.evaluate_gradient(x).flatten()
            y = x - 1 / self.L * grad
            z = z - (epoch + 1) / (2 * self.L) * grad
            x = (epoch + 1) / (epoch + 3) * y + 2 / (epoch + 3) * z

            # Check termination
            loss = self.objective.evaluate_loss(y)

            # If polynomial is psi-approximately vanishing, terminate.
            if loss < self.psi:
                break
            if epoch > 1 and (loss + self.psi*self.tol > loss_list[-1]):
                iter_no_change += 1
                if iter_no_change == self.n_iter_no_change:
                    break

            else:
                iter_no_change = 0
            loss_list.append(loss)

        loss_list.append(self.objective.evaluate_loss(y))
        return y, loss_list, None
