import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd


class ConditionalGradients:
    """The Conditional Gradients algorithm.

    The algorithm will terminate if finding a psi-approximate vanishing polynomial is impossible or as soon as one
    is found.

    Args:
        objective_function: instance of L2Loss
        feasible_region: instance of L1Ball or L2Ball
        oracle_type: str, Optional
            Options are chosen from ["CG", "PCG", "BPCG"]. (Default is "CG".)
        psi: float, Optional
            (Default is 0.1.)
        eps: float, Optional
            (Default is 0.001.)
        max_iterations: int, Optional
            Maximum number of iterations. (Default is 1000.)
        tol: float, Optional
            If no improvement in loss over the last iteration of at least psi * tol is made, terminates the algorithm.
            (Default is 0.0001.)
        compute_loss: bool, Optional
            If true, computes loss and uses it as a termination criterion. (Default is True.)
        inverse_hessian_boost: str, Optional
            Whether or not to use Inverse Hessian Boosting. Here, we can only choose from "full" or "false".
            (Default is "false".)

    Methods:
        check_termination()
            Checks whether self.optimize() has to terminate or not.
        delete_away_vertex()
            Deletes the away_vertex and updates alphas and active_vertices.
        optimize()
            Performs correct Conditional Gradients version.
        optimize_cg()
            Performs the vanilla Conditional Gradients algorithm.
        optimize_pcg()
            Performs the Pairwise Conditional Gradients algorithm.
        optimize_pcg()
            Performs the Blended Pairwise Conditional Gradients algorithm.
    """

    def __init__(self,
                 objective_function,
                 feasible_region,
                 oracle_type: str = "CG",
                 psi: float = 0.1,
                 eps: float = 0.001,
                 max_iterations: int = 1000,
                 tol: float = 0.0001,
                 compute_loss: bool = True,
                 inverse_hessian_boost: str = "false"):
        self.objective = objective_function
        self.region = feasible_region
        self.oracle_type = oracle_type
        self.psi = psi
        self.eps = eps
        self.max_iterations = max_iterations
        self.tol = tol
        self.compute_loss = compute_loss

        # Initialization of algorithm
        self.inverse_hessian_boost = inverse_hessian_boost
        if self.inverse_hessian_boost in ["full"]:
            self.iterate = self.region.projection(self.objective.solution)
        else:
            self.iterate = self.region.initial_vertex()

        self.grad = None
        self.active_vertices = fd(self.iterate)
        self.alphas = cp.array([[1.0]])
        self.fw_gap = 1e16
        self.loss = 1e16
        self.fw_gap_list = [self.fw_gap]
        self.loss_list = [self.loss]
        self.terminate = False

        self.step_size = 1

        # CG
        self.fw_vertex = None
        self.fw_dir = None

        # PCG
        self.away_vertex, self.away_index = None, None
        self.pw_dir = None

        # BPCG
        self.local_fw_vertex, self.local_fw_index = None, None
        self.local_dir = None
        self.final_dir = None

    def check_termination(self):
        """Checks whether self.optimize() has to terminate or not."""

        # If Frank-Wolfe gap is smaller than eps, terminate.
        if self.fw_gap < self.eps:
            self.terminate = True

        # If no improvement is made, terminate.100000,
        elif self.step_size <= 0:
            self.terminate = True

        elif (self.fw_vertex == 0).all():
            self.terminate = True

        elif (self.grad == 0).all():
            self.terminate = True

        elif self.compute_loss:
            self.loss = self.objective.evaluate_function(self.iterate)
            if self.loss + self.psi * self.tol > self.loss_list[-1]:
                self.terminate = True

            # If polynomial is psi-approximate vanishing, terminate.
            elif self.loss < self.psi:
                self.terminate = True

            # If it is no longer possible to create a psi-approximate vanishing polynomial, terminate.
            elif self.loss - self.fw_gap > self.psi:
                self.terminate = True

            self.loss_list.append(self.loss)

    def delete_away_vertex(self):
        """Deletes the away_vertex and updates alphas and active_vertices."""
        if self.away_index == 0:
            self.alphas = self.alphas[1:, :]
            self.active_vertices = self.active_vertices[:, 1:]
        elif self.away_index == self.alphas.shape[0] - 1:
            self.alphas = self.alphas[:-1, :]
            self.active_vertices = self.active_vertices[:, -1:]
        else:
            self.alphas = cp.vstack((self.alphas[:self.away_index, :], self.alphas[self.away_index + 1:, :]))
            self.active_vertices = cp.hstack((self.active_vertices[:, :self.away_index],
                                              self.active_vertices[:, self.away_index + 1:]))

    def optimize(self):
        """Performs correct Conditional Gradients version."""
        if self.oracle_type == "CG":
            self.iterate, self.loss_list, self.fw_gap_list = self.optimize_cg()
        elif self.oracle_type == "PCG":
            self.iterate, self.loss_list, self.fw_gap_list = self.optimize_pcg()
        elif self.oracle_type == "BPCG":
            self.iterate, self.loss_list, self.fw_gap_list = self.optimize_bpcg()
        return self.iterate, self.loss_list, self.fw_gap_list

    def optimize_cg(self):
        """Performs the vanilla Conditional Gradients algorithm.

        Returns:
            self.iterate: cp.ndarray
                The final iterate.
            self.loss_list: list
                The list of all losses.
            self.fw_gap_list
                The list of all fw_gaps.

        References:
            [1] Frank, M. and Wolfe, P. (1956). An algorithm for quadratic programming. Naval research logistics
            quarterly, 3(1-2):95–110.
        """
        for epoch in range(1, self.max_iterations):

            self.grad = self.objective.evaluate_gradient(self.iterate)

            # Frank-Wolfe vertex
            self.fw_vertex, self.fw_gap, _, _ = self.region.linear_minimization_oracle(self.iterate, self.grad)
            self.fw_dir = self.fw_vertex.flatten() - self.iterate.flatten()
            self.fw_gap_list.append(self.fw_gap)

            self.check_termination()
            if self.terminate:
                break

            # Compute step size where the max step size depends on the entry of alpha corresponding to the away vertex.
            self.step_size = self.objective.evaluate_step_size(fd(self.iterate), fd(self.grad), fd(self.fw_dir))

            # Update iterate
            self.iterate = (self.iterate + self.step_size * self.fw_dir).flatten()

        self.loss_list.append(self.objective.evaluate_function(self.iterate))
        return self.iterate, self.loss_list, self.fw_gap_list

    def optimize_pcg(self):
        """Performs the Pairwise Conditional Gradients algorithm.

        Returns:
            self.iterate: cp.ndarray
                The final iterate.
            self.loss_list: list
                The list of all losses.
            self.fw_gap_list
                The list of all fw_gaps.

        References:
            [1] "Lacoste-Julien, S. and Jaggi, M., 2015. On the global linear convergence of Conditional Gradients
            optimization variants. arXiv preprint arXiv:1511.05932."
        """
        for epoch in range(1, self.max_iterations + 1):
            self.grad = self.objective.evaluate_gradient(self.iterate)

            # Frank-Wolfe vertex
            self.fw_vertex, self.fw_gap, fw_index, fw_sign = self.region.linear_minimization_oracle(
                self.iterate, self.grad)
            self.fw_gap_list.append(self.fw_gap)

            self.check_termination()
            if self.terminate:
                break

            # Away vertex
            self.away_vertex, self.away_index = self.region.away_oracle(self.active_vertices, self.grad)

            # Pairwise direction
            self.pw_dir = self.fw_vertex - self.away_vertex

            # Compute step size where the max step size depends on the entry of alpha corresponding to the away vertex.
            max_step_size = float(self.alphas[self.away_index])

            self.step_size = self.objective.evaluate_step_size(
                self.iterate, self.grad, self.pw_dir, max_step_size=max_step_size)

            # Perform Pairwise step.
            # Then, alphas[away_index, 0] = 0. We have to delete the away_vertex and update the fw_vertex.
            if self.step_size == max_step_size:
                self.delete_away_vertex()
            else:
                self.alphas[self.away_index, 0] = float(self.alphas[self.away_index, 0] - self.step_size)
            fw_vertex_active_index = self.region.vertex_among_active_vertices(self.active_vertices, self.fw_vertex)
            if fw_vertex_active_index is None:
                self.alphas = cp.vstack((self.alphas, cp.array([self.step_size])))
                self.active_vertices = cp.hstack((self.active_vertices, fd(self.fw_vertex)))
            else:
                self.alphas[fw_vertex_active_index, 0] += self.step_size

            # Update iterate
            self.iterate = self.iterate + self.step_size * self.pw_dir

        self.loss_list.append(self.objective.evaluate_function(self.iterate))
        return self.iterate, self.loss_list, self.fw_gap_list

    def optimize_bpcg(self):
        """Performs the Blended Pairwise Conditional Gradients algorithm.

        Returns:
            self.iterate: cp.ndarray
                The final iterate.
            self.loss_list: list
                The list of all losses.
            self.fw_gap_list
                The list of all fw_gaps.

        References:
            [1] Tsuji, K., Tanaka, K., and Pokutta, S. (2021). Sparser kernel herding with pairwise conditional
            gradients without swap steps. arXiv preprint arXiv:2110.12650.
        """
        for epoch in range(1, self.max_iterations + 1):
            self.grad = self.objective.evaluate_gradient(self.iterate)

            # Away vertex
            self.away_vertex, self.away_index = self.region.away_oracle(self.active_vertices, self.grad)

            # Local vertex
            self.local_fw_vertex, self.local_fw_index = self.region.local_linear_minimization_oracle(
                self.active_vertices, self.grad)

            # Local direction
            self.local_dir = self.local_fw_vertex.flatten() - self.away_vertex.flatten()

            # Frank-Wolfe vertex
            self.fw_vertex, self.fw_gap, fw_index, fw_sign = self.region.linear_minimization_oracle(
                self.iterate, self.grad)
            self.fw_gap_list.append(self.fw_gap)

            self.check_termination()
            if self.terminate:
                break

            # Global/CG direction
            self.fw_dir = self.fw_vertex.flatten() - self.iterate.flatten()

            # Local step
            if fd(self.grad).T.dot(fd(self.fw_dir)) >= fd(self.grad).T.dot(fd(self.local_dir)):
                self.final_dir = self.local_dir
                # Compute step size where the max step size depends on the entry of alpha corresponding to the away
                # vertex.
                max_step_size = float(self.alphas[self.away_index])
                self.step_size = self.objective.evaluate_step_size(
                    self.iterate, self.grad, self.final_dir, max_step_size=max_step_size)

                # Update local FW vertex
                self.alphas[self.local_fw_index, 0] += self.step_size
                if self.step_size == max_step_size:
                    self.delete_away_vertex()
                else:
                    self.alphas[self.away_index, 0] = float(self.alphas[self.away_index, 0] - self.step_size)
            # Global step
            else:
                self.final_dir = self.fw_dir
                max_step_size = 1
                self.step_size = self.objective.evaluate_step_size(
                    self.iterate, self.grad, self.final_dir, max_step_size=max_step_size)
                # Update convex decomposition
                if self.step_size == max_step_size:
                    self.active_vertices = fd(self.fw_vertex)
                    self.alphas = cp.array([[1.0]])
                else:
                    self.alphas[:, 0] = (1 - self.step_size) * self.alphas[:, 0]
                    fw_vertex_active_index = self.region.vertex_among_active_vertices(self.active_vertices,
                                                                                      self.fw_vertex)

                    # Check if FW vertex is already an active vertex
                    if fw_vertex_active_index is None:
                        self.alphas = cp.vstack((self.alphas, cp.array([self.step_size])))
                        self.active_vertices = cp.hstack((self.active_vertices, fd(self.fw_vertex)))
                    else:
                        self.alphas[fw_vertex_active_index, 0] += self.step_size

            # Update iterate
            self.iterate = self.iterate + self.step_size * self.final_dir

        self.loss_list.append(self.objective.evaluate_function(self.iterate))
        return self.iterate, self.loss_list, self.fw_gap_list
