import random
import unittest
import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.oracles.accelerated_gradient_descent import AcceleratedGradientDescent
from src.oracles.conditional_gradients import ConditionalGradients
from src.oracles.feasible_regions import L1Ball, L2Ball
from src.oracles.objective_functions import L2Loss


class TestL1Ball(unittest.TestCase):
    def setUp(self):
        self.feasiblityRegion = L1Ball(5, 2)

    def test_initial_vertex(self):
        """Tests whether L1Ball.initial_vertex() behaves as intended."""
        inital_vertex = self.feasiblityRegion.initial_vertex()
        self.assertTrue((inital_vertex == cp.array([2, 0, 0, 0, 0])).all(), "Initial vertex is not correctly created.")

    def test_away_oracle(self):
        """Tests whether L1Ball.away_oracle() behaves as intended."""
        active_vertices = cp.array([[1, 0], [0.8, 0.811], [0.8, 0], [0.7, 0], [0.6, 0]])
        direction = cp.array([0, 1, 0, 0, 0])
        away_vertex, active_vertices_idx = self.feasiblityRegion.away_oracle(active_vertices, direction)
        self.assertTrue((away_vertex == cp.array([0, 0.811, 0, 0, 0])).all(), "Wrong away_vertex returned.")
        self.assertTrue(active_vertices_idx == 1, "Wrong active_vertices_index returned.")

    def test_linear_minimization_oracle(self):
        """Tests whether L1Ball.linear_minimization_oracle() behaves as intended."""
        direction = cp.array([0, 1, 0, 0, 0])
        x = cp.array([0.1, 0.2, 0.3, 0.4, 0.5])
        fw_vertex, fw_gap, non_zero_idx, sign = self.feasiblityRegion.linear_minimization_oracle(x, direction)
        self.assertTrue((fw_vertex == cp.array([0, -2, 0, 0, 0])).all(), "Wrong fw_vertex returned.")
        self.assertTrue(sign == -1, "Wrong sign returned.")
        self.assertTrue(non_zero_idx == 1, "Wrong non_zero_idx returned.")
        self.assertTrue(fw_gap == 2.2, "Wrong fw_gap computed.")
        self.assertTrue(fw_vertex.T.dot(direction) <= 0, "<fw_vertex, direction> has to be nonnegative.")

    def test_vertex_among_active_vertices(self):
        """Tests whether L1Ball.vertex_among_active_vertices() behaves as intended."""
        matrix = cp.array([[1, -1, 0],
                           [0, 0, 1]])
        vector = cp.array([[0], [-2]])
        self.assertTrue(self.feasiblityRegion.vertex_among_active_vertices(matrix, vector) is None,
                        "Did not recognize that vertex is not active yet.")
        vector = cp.array([[-1], [0]])
        self.assertTrue(self.feasiblityRegion.vertex_among_active_vertices(matrix, vector) == 1,
                        "Wrong index returned.")
        vector = cp.array([[0], [1]])
        self.assertTrue(self.feasiblityRegion.vertex_among_active_vertices(matrix, vector) == 2,
                        "Wrong index returned.")
        matrix = cp.array([[1]])
        vector = cp.array([[1]])
        self.assertTrue(self.feasiblityRegion.vertex_among_active_vertices(matrix, vector) == 0,
                        "Wrong index returned.")

    def test_projection(self):
        """Tests whether L1Ball.projection() behaves asa intended."""
        region = L1Ball(3, 2)
        for i in range(0, 10):
            vector_to_project = 2 * (cp.random.random((3, 1)) - 1) / 2
            scaling_factor = (i % 5)
            vector_to_project = vector_to_project * scaling_factor
            vector_projected = self.feasiblityRegion.projection(vector_to_project).flatten()

            eps = 10 ** (-10)
            psi = 0
            iterations = 1000
            A = cp.identity(3)
            b = -vector_to_project
            lmbda = 0.0
            objective = L2Loss(A, b, lmbda)
            oracle = ConditionalGradients(objective_function=objective, feasible_region=region, oracle_type="BPCG",
                                          psi=psi,
                                          eps=eps, max_iterations=iterations, inverse_hessian_boost="false",
                                          compute_loss=False)

            iterate, _, list_a = oracle.optimize()
            iterate = iterate.flatten()
            self.assertTrue(cp.linalg.norm(iterate - vector_projected) <= 1 / iterations,
                            "The projection operation is wrong.")
            self.assertTrue(cp.linalg.norm(iterate, ord=1) - 2 <= 10**(-6), "The projection operation is wrong.")


class TestL2Ball(unittest.TestCase):
    def setUp(self):
        self.feasiblityRegion = L2Ball(5, 2)

    def test_initial_vertex(self):
        """Tests whether L2Ball.initial_vertex() behaves as intended."""
        inital_vertex = self.feasiblityRegion.initial_vertex()
        self.assertTrue((inital_vertex == cp.array([2, 0, 0, 0, 0])).all(), "Initial vertex is not correctly created.")

    def test_linear_minimization_oracle(self):
        """Tests whether L2Ball.linear_minimization_oracle() behaves as intended."""
        direction = cp.array([0, 1, 0, 0, 0])
        x = cp.array([1, 0, 0, 0, 0])
        fw_vertex, fw_gap, _, _ = self.feasiblityRegion.linear_minimization_oracle(x, direction)
        self.assertTrue((fw_vertex == cp.array([0, -2, 0, 0, 0])).all(), "Wrong fw_vertex returned.")
        self.assertTrue(fw_gap == 2.0, "Wrong fw_gap computed.")
        self.assertTrue(fw_vertex.T.dot(direction) <= 0, "<fw_vertex, direction> has to be nonnegative.")


class TestL2Loss(unittest.TestCase):

    def test_precomputations(self):
        """Tests whether precomputed matrices are treated correctly."""
        A = cp.array([[1, 1],
                      [1, 1],
                      [1, 0]])

        m, n = A.shape

        A_inv = cp.array([[1., -1.],
                          [-1., 1.5]])

        b = cp.array([[1],
                      [0],
                      [0]])

        b_squared = float(b.T.dot(b))

        A_squared = A.T.dot(A)
        A_b = A.T.dot(fd(b))

        lmbda = 1.0

        objective_function = L2Loss(A, b, lmbda=lmbda)
        self.assertAlmostEqual(float(abs((2 / m) * b_squared - objective_function.b_squared)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(float(cp.linalg.norm((2 / m) * A_b - objective_function.A_b)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(
            float(cp.linalg.norm((2 / m) * A_squared + lmbda * cp.identity(n) - objective_function.A_squared)), 0,
            7, "Error in L2Loss")

        objective_function = L2Loss(A, b, lmbda=lmbda, data_squared=A_squared, data_labels=A_b,
                                    labels_squared=b_squared)

        self.assertAlmostEqual(float(abs((2 / m) * b_squared - objective_function.b_squared)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(float(cp.linalg.norm((2 / m) * A_b - objective_function.A_b)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(
            float(cp.linalg.norm((2 / m) * A_squared + lmbda * cp.identity(n) - objective_function.A_squared)), 0,
            7, "Error in L2Loss")

        objective_function = L2Loss(A, b, data_squared=A_squared, data_labels=A_b,
                                    data_squared_inverse=A_inv)

        self.assertAlmostEqual(float(abs((2 / m) * b_squared - objective_function.b_squared)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(float(cp.linalg.norm((2 / m) * A_b - objective_function.A_b)), 0, 7, "Error in L2Loss")
        self.assertAlmostEqual(
            float(cp.linalg.norm((2 / m) * A_squared - objective_function.A_squared)), 0,
            7, "Error in L2Loss")

        self.assertAlmostEqual(
            float(cp.linalg.norm((m / 2) * A_inv - objective_function.A_squared_inv)), 0,
            7, "Error in L2Loss")

    def test_evaluate_function(self):
        """Tests whether L2Loss.evaluate_function() behaves as intended."""
        for m in range(1, 5):
            for n in range(1, 10):
                A = cp.random.random((m, n))
                b = cp.random.random((m, 1)).flatten()
                x = cp.random.random((n, 1)).flatten()
                lmbda = float(random.random() * n)
                objective_function = L2Loss(A, b, lmbda=lmbda)
                self.assertTrue(
                    abs(1 / A.shape[0] * cp.linalg.norm(A.dot(fd(x)).flatten() + b) ** 2 + lmbda * cp.linalg.norm(
                        x) ** 2 / 2 - objective_function.evaluate_function(x)) <= 1e-10, "Wrong loss returned.")

    def test_evaluate_gradient(self):
        """Tests whether L2Loss.evaluate_gradient() behaves as intended."""
        for m in range(1, 5):
            for n in range(1, 10):
                A = cp.random.random((m, n))
                b = cp.random.random((m, 1)).flatten()
                x = cp.random.random((n, 1)).flatten()
                lmbda = float(random.random() * n)
                objective_function = L2Loss(A, b, lmbda=lmbda)
                gradient = 2 / m * (A.T.dot(A).dot(x) + A.T.dot(b) + m / 2 * lmbda * x)
                self.assertTrue(
                    (abs(gradient.flatten() - objective_function.evaluate_gradient(x).flatten()) <= 1e-10).all(),
                    "Wrong gradient returned.")

    def test_L(self):
        """Tests whether L2Loss.L() behaves as intended."""
        data_matrix = cp.random.random((5, 3))
        labels = cp.random.random((5, 1))
        loss = L2Loss(data=data_matrix, labels=labels, lmbda=1)
        L = loss.L()
        L_computed = 2 / data_matrix.shape[0] * cp.max(cp.linalg.eigh(data_matrix.T.dot(data_matrix))[0]) + 1
        self.assertTrue(abs(float(L - L_computed)) <= 1e-10, "L is wrong.")

    def test_evaluate_step_size(self):
        """Tests whether L2Loss.evaluate_step_size() behaves as intended."""
        for m in range(1, 4):
            for n in range(1, 4):
                A = cp.random.random((m, n))
                b = cp.random.random((m, 1)).flatten()
                x = cp.random.random((n, 1)).flatten()
                lmbda = float(random.random() * 7)
                objective_function = L2Loss(A, b, lmbda=lmbda)
                feasible_region = L1Ball(n, 5)
                gradient = objective_function.evaluate_gradient(x)
                fw_vertex, _, _, _ = feasible_region.linear_minimization_oracle(x, gradient)
                direction = fw_vertex - x
                exact = objective_function.evaluate_step_size(x, gradient, direction, step_size_rule="exact")
                ls = objective_function.evaluate_step_size(x, gradient, direction, step_size_rule="line_search",
                                                           iterations_line_search=1000)
                tmp_exact = x + exact * direction
                current_value_exact = objective_function.evaluate_function(tmp_exact)
                tmp_ls = x + ls * direction
                current_value_ls = objective_function.evaluate_function(tmp_ls)
                self.assertTrue(abs(ls - exact) <= 1e-3, "Exact and line search provide different results.")
                self.assertTrue(abs(
                    current_value_ls - current_value_exact) <= 1e-4, "Exact and line search provide different results.")


class TestMinimizationAlgorithms(unittest.TestCase):

    def test_pairwise_frank_wolfe(self):
        """Tests whether ConditionalGradients behaves as intended."""
        for i in range(5):
            m = random.randint(1, 1000)
            n = random.randint(1, 1000)
            radius = random.random() * 100 + 1
            psi = 0.0
            eps = 0.01
            A = cp.random.random((m, n))
            b = cp.random.random((m, 1))
            lmbda = 0.0
            objective = L2Loss(A, b, lmbda)
            region = L1Ball(n, radius)
            oracle = ConditionalGradients(objective_function=objective, feasible_region=region, oracle_type="PCG",
                                          psi=psi,
                                          eps=eps, max_iterations=1000, inverse_hessian_boost="false")
            iterate, loss_list, fw_gaps = oracle.optimize()
            self.assertTrue((len(iterate) == n), "Error.")

    def test_blended_pairwise_frank_wolfe(self):
        """Tests whether ConditionalGradients behaves as intended."""
        for i in range(5):
            m = random.randint(1, 1000)
            n = random.randint(1, 1000)
            radius = random.random() * 100 + 1
            psi = 0.0
            eps = 0.01
            A = cp.random.random((m, n))
            b = cp.random.random((m, 1))
            lmbda = 0.0
            objective = L2Loss(A, b, lmbda)
            region = L1Ball(n, radius)
            oracle = ConditionalGradients(objective_function=objective, feasible_region=region, oracle_type="BPCG",
                                          psi=psi,
                                          eps=eps, max_iterations=1000, inverse_hessian_boost="false")

            iterate, loss_list, fw_gaps = oracle.optimize()
            self.assertTrue((len(iterate) == n), "Error.")

    def test_vanilla_frank_wolfe(self):
        """Tests whether ConditionalGradients behaves as intended."""
        for i in range(0, 5):
            m = random.randint(1, 1000)
            n = random.randint(1, 1000)
            radius = random.random() * 100 + 1
            psi = 0.0
            eps = 0.01
            A = cp.random.random((m, n))
            b = cp.random.random((m, 1))
            lmbda = 0.0
            objective = L2Loss(A, b, lmbda)
            region = L1Ball(n, radius)
            oracle = ConditionalGradients(objective_function=objective, feasible_region=region, oracle_type="CG",
                                          psi=psi,
                                          eps=eps, max_iterations=1000, inverse_hessian_boost="false")
            iterate, loss_list, fw_gaps = oracle.optimize()
            self.assertTrue((len(iterate) == n), "Error.")

    def test_accelerated_gradient_descent(self):
        """Tests whether AcceleratedGradientDescent behaves as intended."""
        for i in range(5):
            m = random.randint(1, 1000)
            n = random.randint(1, 1000)
            A = cp.random.random((m, n))
            b = cp.random.random((m, 1))
            lmbda = random.random() * 10
            objective = L2Loss(A, b, lmbda)
            oracle = AcceleratedGradientDescent(objective_function=objective, dimension=n, max_iterations=1000,
                                                inverse_hessian_boost="false")
            iterate, loss_list, fw_gaps = oracle.optimize()
            self.assertTrue((len(iterate) == n), "Error.")
