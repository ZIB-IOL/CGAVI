import cupy as cp

from src.auxiliary_functions.auxiliary_functions import fd
from src.oracles.feasibility_regions import L1Ball
from src.oracles.objective_functions import L2Loss


class FrankWolfe:
    """
    The Frank-Wolfe algorithm class.

    The algorithm will terminate if finding a psi approximately vanishing polynomial is impossible or as soon as one
    is found.

    Args:
        objective_function: instance of L2Loss
        feasibility_region: instance of L1Ball or L2Ball
        psi: float, Optional
            (Default is 0.1.)
        eps: float, Optional
            (Default is 0.1.)
        max_iterations: int, Optional
            Maximum number of iterations. (Default is 1000.)
        tol: float, Optional
            If no improvement in loss over the last iteration of at least psi * tol is made, terminate the algorithm.
        compute_loss: bool, Optional
            If true, computes loss and uses it as termination criterion. (Default is True.)

    Methods:
        check_termination(epoch)
            Checks whether self.optimize() has to terminate or not.
        delete_away_vertex()
            Deletes the away_vertex and updates alphas and active_vertices.
        optimize()
            Performs correct Frank-Wolfe version.
        optimize_vanilla()
            Performs vanilla Frank-Wolfe.
        optimize_pfw()
            Performs Pairwise Frank-Wolfe.

    References:
        "Lacoste-Julien, S. and Jaggi, M., 2015. On the global linear convergence of Frank-Wolfe optimization
        variants. arXiv preprint arXiv:1511.05932."
    """

    def __init__(self,
                 objective_function,
                 feasibility_region,
                 psi: float = 0.1,
                 eps: float = 0.1,
                 max_iterations: int = 1000,
                 tol: float = 0.0005,
                 compute_loss: bool = True):
        self.objective = objective_function
        self.region = feasibility_region
        self.psi = psi
        self.eps = eps
        self.max_iterations = max_iterations
        self.tol = tol
        self.compute_loss = compute_loss

        # Initialization of algorithm
        self.iterate = self.region.initial_vertex()
        self.active_vertices = fd(self.iterate)
        self.alphas = cp.array([[1.0]])
        self.fw_gap = 1e16
        self.loss = 1e16
        self.fw_gap_list = [self.fw_gap]
        self.loss_list = [self.loss]
        self.terminate = False

        self.fw_vertex = None
        self.away_vertex, self.away_index = None, None
        self.fw_dir = None
        self.pw_dir = None
        self.step_size = 1

    def check_termination(self):
        """Checks whether self.optimize() has to terminate or not."""

        # If Frank-Wolfe gap is smaller than eps, terminate.
        if self.fw_gap < self.eps:
            self.terminate = True

        # If no improvement is made, terminate.
        elif self.step_size <= 0:
            self.terminate = True

        elif (self.fw_vertex == 0).all():
            self.terminate = True

        elif self.compute_loss:
            self.loss = self.objective.evaluate_loss(self.iterate)
            if self.loss > self.loss_list[-1]:
                print("What?")
            if self.loss + self.psi * self.tol > self.loss_list[-1]:
                self.terminate = True
            self.loss_list.append(self.loss)

            # If polynomial is psi-approximately vanishing, terminate.
            if self.loss < self.psi:
                self.terminate = True

            # If it is no longer possible to create a psi-approximately vanishing polynomial, terminate.
            elif self.loss - self.fw_gap > self.psi:
                self.terminate = True

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
        """Performs correct Frank-Wolfe version."""
        if self.region.name() == "l1":
            self.iterate, self.loss_list, self.fw_gap_list = self.optimize_pfw()
        elif self.region.name() == "l2":
            self.iterate, self.loss_list, self.fw_gap_list = self.optimize_vanilla()

        return self.iterate, self.loss_list, self.fw_gap_list

    def optimize_vanilla(self):
        """Performs Vanilla Frank-Wolfe.

        Returns:
            self.iterate: cp.ndarray
                The final iterate.
            self.loss_list: list
                The list of all losses.
            self.fw_gap_list
                The list of all fw_gaps.
            """

        for epoch in range(1, self.max_iterations):

            grad = self.objective.evaluate_gradient(self.iterate)

            # Frank-Wolfe vertex
            self.fw_vertex, self.fw_gap = self.region.linear_minimization_oracle(self.iterate, grad)
            self.fw_dir = self.fw_vertex.flatten() - self.iterate.flatten()
            self.fw_gap_list.append(self.fw_gap)

            self.check_termination()
            if self.terminate:
                break

            # Compute step size where the max step size depends on the entry of alpha corresponding to the away vertex.
            self.step_size = self.objective.evaluate_step_size(fd(self.iterate), fd(grad), fd(self.fw_dir))

            # Update iterate
            self.iterate = (self.iterate + self.step_size * self.fw_dir).flatten()
        self.loss_list.append(self.objective.evaluate_loss(self.iterate))
        return self.iterate, self.loss_list, self.fw_gap_list

    def optimize_pfw(self):
        """Performs Pairwise Frank-Wolfe.

        Returns:
            self.iterate: cp.ndarray
                The final iterate.
            self.loss_list: list
                The list of all losses.
            self.fw_gap_list
                The list of all fw_gaps.
            """

        for epoch in range(1, self.max_iterations):
            grad = self.objective.evaluate_gradient(self.iterate)

            # Frank-Wolfe vertex
            self.fw_vertex, fw_index, fw_sign, self.fw_gap = self.region.linear_minimization_oracle(
                self.iterate, grad)
            self.fw_gap_list.append(self.fw_gap)

            self.check_termination()
            if self.terminate:
                break

            # Away vertex
            self.away_vertex, self.away_index = self.region.away_oracle(self.active_vertices, grad)

            # Pairwise direction
            self.pw_dir = self.fw_vertex - self.away_vertex

            # Compute step size where the max step size depends on the entry of alpha corresponding to the away vertex.
            max_step_size = float(self.alphas[self.away_index])

            self.step_size = self.objective.evaluate_step_size(
                self.iterate, grad, self.pw_dir, max_step_size=max_step_size)

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
        self.loss_list.append(self.objective.evaluate_loss(self.iterate))
        return self.iterate, self.loss_list, self.fw_gap_list
