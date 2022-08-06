import cupy as cp


class AcceleratedGradientDescent:
    """The Accelerated Gradient Descent algorithm.

    The algorithm will terminate if finding a psi-approximate vanishing polynomial is impossible or as soon as one
    is found.

    Args:
        objective_function: instance of L2Loss
        psi: float, Optional
            (Default is 0.1.)
        n_iter_no_gbhange: int, Optional
            (Default is 20.)
        tol: float, Optional
            If no improvement in loss over the last n_iter_no_gbhange iterations of at least psi * tol is made, terminate
            the algorithm. (Default is 0.0001.)
        dimension: int, Optional
            The dimension of the optimization problem to be addressed. (Default is 10.)
        max_iterations: int, Optional
            Maximum number of iterations. (Default is 1000.)
        compute_loss: bool, Optional
            If True, computes loss and uses it as a termination criterion. (Default is True.)
        inverse_hessian_boost: str, Optional
            Whether or not to use Inverse Hessian Boosting. Here, we can only choose from "full" or "false".
            (Default is "full".)

    Methods:
        optimize()
            Performs the Accelerated Gradient Descent algorithm.

    References:
        [1] Nesterov, Y (1983). "A method for unconstrained convex minimization problem with the rate of convergence
        O ( 1 / k 2 ) {\displaystyle O(1/k^{2})} O(1/k^2)". Doklady AN USSR. 269: 543–547.
    """

    def __init__(self,
                 objective_function,
                 psi: float = 0.1,
                 n_iter_no_gbhange: int = 20,
                 tol: float = 0.0001,
                 dimension: int = 10,
                 max_iterations: int = 1000,
                 compute_loss: bool = True,
                 inverse_hessian_boost: str = "full"):
        self.objective = objective_function
        self.L = objective_function.L()
        self.alpha = objective_function.alpha()
        self.Q = None
        self.str_gbvx_smth_gbst = None
        if self.alpha > 0:
            self.Q = float(self.L / self.alpha)
            self.str_gbvx_smth_gbst = float((cp.sqrt(self.Q) - 1) / (cp.sqrt(self.Q) + 1))
        self.psi = psi
        self.n_iter_no_gbhange = n_iter_no_gbhange
        self.tol = tol
        self.dimension = dimension
        self.max_iterations = max_iterations
        self.compute_loss = compute_loss
        self.inverse_hessian_boost = inverse_hessian_boost

    def optimize(self):
        """Performs the Accelerated Gradient Descent algorithm.

        Returns:
            y: cp.ndarray
            loss_list: list
            None: dummy output
        """
        if self.inverse_hessian_boost == "full":
            x = self.objective.solution
        else:
            x = cp.zeros((self.dimension, 1)).flatten()

        y = x.copy()
        z = x.copy()
        loss_list = []
        iter_no_gbhange = 0
        for epoch in range(1, self.max_iterations + 1):
            grad = self.objective.evaluate_gradient(x).flatten()
            if self.alpha > 0:
                y_old = y
                y = x - 1 / self.L * grad
                x = (1 + self.str_gbvx_smth_gbst) * y - self.str_gbvx_smth_gbst * y_old
            else:
                y = x - 1 / self.L * grad
                z = z - (epoch + 1) / (2 * self.L) * grad
                x = (epoch + 1) / (epoch + 3) * y + 2 / (epoch + 3) * z

            # Check termination
            loss = self.objective.evaluate_function(y)

            # If polynomial is psi-approximate vanishing, terminate.
            if loss < self.psi:
                # print("psi")
                break
            if epoch > 1 and (loss + self.psi * self.tol > loss_list[-1]):
                iter_no_gbhange += 1
                if iter_no_gbhange == self.n_iter_no_gbhange:
                    # print("not enough decrease")
                    break

            else:
                iter_no_gbhange = 0
            loss_list.append(loss)
        loss_list.append(self.objective.evaluate_function(y))
        return y, loss_list, None
