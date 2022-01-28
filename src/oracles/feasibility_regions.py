import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.auxiliary_functions.indices import get_non_zero_indices


class L1Ball:
    """A class used to represent the l1 ball.

    Args:
        dimension: integer
            The dimension of the ball.
        radius: float, Optional
            The radius of the ball. (Default is 1.0)

    Methods:
        name()
            Returns the name of the region.
        initial_vertex()
            Returns the initial vertex.
        away_oracle(active_vertices: cp.ndarray, direction: cp.ndarray)
            Solves the maximization problem max_{i} in Lp <direction, active_vertices[:, i]>.
        linear_minimization_oracle(x: cp.ndarray, grad: cp.ndarray)
            Solves the linear minimization problem min_g in Lp <grad, y>.
    """

    def __init__(self, dimension: int, radius: float = 1.0):
        self.dimension = int(dimension)
        self.radius = radius
        assert self.radius > 1, "Radius needs to be greater than 1."

    def name(self):
        """Returns the name of the region."""
        return 'l1'

    def initial_vertex(self):
        """Returns the initial vertex."""
        x = cp.zeros(self.dimension)
        x[0] = self.radius
        return x

    def away_oracle(self, active_vertices: cp.ndarray, direction: cp.ndarray):
        """Solves the maximization problem max_{i} in Lp <direction, active_vertices[:, i]>.

        Args:
            active_vertices: cp.ndarray
                A matrix whose column vectors are vertices of the l1 ball.
            direction: cp.ndarray
                Gradient at x.

        Returns:
            active_vertices_idx: int
                Reference to the column in the active set corresponding to the away vertex.
            away_vertex: cp.ndarray
                The away vertex.
        """
        tmp = active_vertices.T.dot(direction)
        active_vertices_idx = cp.argmax(tmp)
        away_vertex = active_vertices[:, active_vertices_idx]
        return away_vertex, active_vertices_idx

    def linear_minimization_oracle(self, x: cp.ndarray, direction: cp.ndarray):
        """Solves the linear minimization problem min_g in Lp <direction, y>.

        Args:
            x: cp.ndarray
                Point in the l1 ball.
            direction: cp.ndarray
                Gradient at x.

        Returns:
            fw_vertex: cp.ndarray
                The solution to the linear minimization problem.
            non_zero_idx: int
                The index where the Frank-Wolfe vertex is not zero.
            sign: int
                The sign of y[idx], either 1 or -1.
            wolfe_gap: float
                The Frank-Wolfe gap.
        """
        non_zero_idx = cp.abs(direction).argmax()
        sign = int(- cp.sign(direction[non_zero_idx]))
        fw_vertex = cp.zeros(self.dimension)
        fw_vertex[non_zero_idx] = sign * self.radius
        fw_gap = float(direction.T.dot(fd(x) - fd(fw_vertex)))
        return fw_vertex, non_zero_idx, sign, fw_gap

    def vertex_among_active_vertices(self, active_vertices: cp.ndarray, fw_vertex: cp.ndarray):
        """Checks if the fw_vertex is in the set of active vertices

        Args:
            active_vertices: cp.ndarray
                A matrix whose column vectors are vertices of the l1 ball.
            fw_vertex: cp.ndarray
                The Frank-Wolfe vertex.

        Returns:
            active_vertex_index:
                Returns the position of fw_vertex in active_vertices as an int. If fw_vertex is not a column of
                active_vertices, this value is None.
        """
        active_vertices = fd(active_vertices)
        index = get_non_zero_indices(fw_vertex)
        assert len(index) == 1, "Vertices should have exactly one non-zero entry."
        index = index[0]
        value = fd(fw_vertex)[index, 0]
        crucial_row = active_vertices[index, :]
        list_of_indices = get_non_zero_indices(crucial_row)
        assert len(list_of_indices) <= 2, "Vertices should not occur twice in active_vertices."
        for active_vertex_index in list_of_indices:
            if crucial_row[active_vertex_index] * value > 0:
                return active_vertex_index
        return None


class L2Ball:
    """A class used to represent the l2 ball.

    Args:
        dimension: integer
            The dimension of the ball.
        radius: float, Optional
            The radius of the ball. (Default is 1.0)

    Methods:
        name()
            Returns the name of the region.
        initial_vertex()
            Returns the initial vertex.
        away_oracle(active_vertices: cp.ndarray, direction: cp.ndarray)
            Solves the maximization problem max_{i} in Lp <direction, active_vertices[:, i]>.
        linear_minimization_oracle(x: cp.ndarray, grad: cp.ndarray)
            Solves the linear minimization problem min_g in Lp <grad, y>.
    """

    def __init__(self, dimension: int, radius: float = 1.0):
        self.dimension = int(dimension)
        self.radius = radius
        assert self.radius > 1, "Radius needs to be greater than 1."

    def name(self):
        """Returns the name of the region."""
        return 'l2'

    def initial_vertex(self):
        """Returns the initial vertex."""
        x = cp.zeros(self.dimension)
        x[0] = self.radius
        return x

    def linear_minimization_oracle(self, x: cp.ndarray, direction: cp.ndarray):
        """Solves the linear minimization problem min_g in Lp <direction, y>.

        Args:
            x: cp.ndarray
                Point in the l1 ball.
            direction: cp.ndarray
                Gradient at x.

        Returns:
            fw_vertex: cp.ndarray
                The solution to the linear minimization problem.
            non_zero_idx: int
                The index where the Frank-Wolfe vertex is not zero.
            sign: int
                The sign of y[idx], either 1 or -1.
            wolfe_gap: float
                The Frank-Wolfe gap.
        """
        fw_vertex = - direction / cp.linalg.norm(direction) * self.radius
        fw_gap = float(direction.T.dot(fd(x) - fd(fw_vertex)))
        return fw_vertex, fw_gap