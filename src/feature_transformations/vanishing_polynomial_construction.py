import cupy as cp
from src.auxiliary_functions.auxiliary_functions import orthogonal_projection, fd
from src.auxiliary_functions.indices import find_first_non_zero_entries, find_last_non_zero_entries
from src.auxiliary_functions.sorting import sort_by_pivot, row_sort
from src.feature_transformations.auxiliary_functions_avi import update_coefficient_vectors


def find_range_null_avi(O_terms: cp.ndarray, O_evaluations: cp.ndarray, border_terms: cp.ndarray,
                        border_evaluations: cp.ndarray, psi: float, tau: float):
    """Performs FindRangeNull (using SVD) for AVI.

    Args:
        O_terms: cp.ndarray
        O_evaluations: cp.ndarray
        border_terms: cp.ndarray
        border_evaluations: cp.ndarray
        psi: float
        tau: float

    Returns:
        coefficient_vectors: cp.ndarray
            New coefficient vectors.
        new_leading_terms: cp.ndarray
            New leading terms.
        new_O_terms: cp.ndarray
            New O terms.
        new_O_evaluations: cp.ndarray
            New evaluations of O terms.
        new_O_indices: list
            New indices corresponding to O terms.

    References:
        [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
    """

    # # Need to flip because srref would otheriwse clean out O terms and not border terms
    O_border_evaluations = cp.hstack((O_evaluations, border_evaluations))
    O_border_evaluations_flipped = cp.flip(cp.hstack((O_evaluations, border_evaluations)), axis=1)

    # Need to flip because srref would otheriwse clean out O terms and not border terms
    B = approximate_onb_algorithm(O_border_evaluations, psi=psi)

    B_flipped = cp.flip(B, axis=0)
    del B
    coefficient_vectors = None
    end_degree = False
    while B_flipped.shape[1] > 0:
        C, indices = stabilized_reduced_row_echelon_form_algorithm(B_flipped.T, tau=tau)
        if coefficient_vectors is None:
            coefficient_vectors = C
        else:
            for i in range(0, C.shape[0]):
                coefficient_vector = fd(C.T[:, i])
                coefficient_vectors_new = update_coefficient_vectors(fd(coefficient_vectors), fd(coefficient_vector),
                                                                     first=True)
                if coefficient_vectors_new.shape == coefficient_vectors.shape:
                    end_degree = True
        lt_indices = find_first_non_zero_entries(coefficient_vectors)
        if lt_indices is None:
            break

        for idx in lt_indices:
            try:
                indices.remove(idx)
            except ValueError:
                pass

        O_border_evaluations_flipped = O_border_evaluations_flipped[:, indices]
        O_border_evaluations = cp.flip(O_border_evaluations_flipped, axis=1)

        B = approximate_onb_algorithm(O_border_evaluations, psi=psi)
        B_flipped = cp.flip(B, axis=0)
        del B
        if end_degree:
            break
    if coefficient_vectors is not None:
        coefficient_vectors = cp.flip(coefficient_vectors.T, axis=0)
        coefficient_vectors = sort_by_pivot(coefficient_vectors)

    coefficient_vectors, new_leading_terms, new_O_terms, new_O_evaluations, new_O_indices = get_info(
        O_terms, border_terms, border_evaluations, coefficient_vectors)
    return coefficient_vectors, new_leading_terms, new_O_terms, new_O_evaluations, new_O_indices


def get_info(O_terms: cp.ndarray, border_terms: cp.ndarray, border_evaluations: cp.ndarray,
             coefficient_vectors: cp.ndarray):
    """Get info that has to be passed on for a subroutine of AVI.

    Returns:
        coefficient_vectors: cp.ndarray
        new_leading_terms: cp.ndarray
        new_O_terms: cp.ndarray
        new_O_evaluations: cp.ndarray
        new_O_indices: list
    """

    if coefficient_vectors is None:
        new_O_indices = [x for x in list(range(0, border_terms.shape[1]))]
        return None, None, border_terms, border_evaluations, new_O_indices
    lt_indices = find_last_non_zero_entries(coefficient_vectors)

    O_indices = [x for x in list(range(0, coefficient_vectors.shape[0])) if x not in lt_indices]
    lt_indices = [x - O_terms.shape[1] for x in lt_indices if x - O_terms.shape[1] >= 0]
    new_O_indices = [x - O_terms.shape[1] for x in O_indices if x - O_terms.shape[1] >= 0]

    for x in lt_indices:
        assert x >= 0, "error"

    new_leading_terms = border_terms[:, lt_indices]
    new_O_terms = border_terms[:, new_O_indices]
    new_O_evaluations = border_evaluations[:, new_O_indices]
    return coefficient_vectors, new_leading_terms, new_O_terms, new_O_evaluations, new_O_indices


def approximate_onb_algorithm(matrix: cp.ndarray, psi: float = 0.1):
    """Performs APONB on matrix.

    References:
        [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
    """
    _, D, V = cp.linalg.svd(matrix.T.dot(matrix))

    idx = len(D)
    for i in range(0, len(D)):
        row_V = fd(V[i, :]).T
        evaluation_vector = fd(matrix.dot(row_V.T))

        loss = (1 / matrix.shape[0]) * cp.linalg.norm(evaluation_vector) ** 2
        if loss < psi:
            idx = i
            break

        # This would be the loop termination criterion from the original paper. However, psi now has to be tuned for
        # every data set individually.
        # if float(cp.sqrt(D[i])) < psi:
        #     idx = i
        #     break
    return cp.array(V.T[:, idx:])


def stabilized_reduced_row_echelon_form_algorithm(matrix: cp.ndarray, tau: float = 0.1):
    """Performs the Stabilized Reduced Row Echelon Form (SRREF) algorithm.

    Brings matrix into SRREF and returns R and the column indices of the pivots.

    Args:
        matrix: cp.ndarray
        tau: float, Optional
            (Default is 0.1.)

    Returns:
        R: cp.ndarray
        indices: list

    References:
        [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
        polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
        [2] Limbeck, J., 2013. Computation of approximate border bases and applications (Doctoral dissertation,
        Universität Passau).
    """

    Q, R = qr_decomposition(matrix, tau=tau)
    R = reduced_row_echelon_form_algorithm(R)

    # Clean out rows above.
    m, n = R.shape
    for i in range(0, m):
        norm = cp.linalg.norm(R[i, :])
        if norm <= tau:
            R[i, :][cp.newaxis, :] = cp.zeros((1, n))
        else:
            R[i, :] = R[i, :] / norm
    R = R[~cp.all(R == 0, axis=1)]
    m, n = R.shape

    # Create list of indices
    indices = []
    for i in range(0, m):
        index_list_as_array = [cp.where(R[i, :][cp.newaxis, :] != 0)[1][0]]
        index_list = [int(x) for x in index_list_as_array]
        indices.extend(index_list)
    return R, indices


def qr_decomposition(matrix: cp.ndarray, tau: float = 0.1):
    """Performs QR decomposition on matrix.

    References:
        [1] Limbeck, J., 2013. Computation of approximate border bases and applications (Doctoral dissertation,
        Universität Passau).
    """
    m, n = matrix.shape
    mn = min(m, n)

    a = matrix[:, 0][:, cp.newaxis]
    norm = cp.linalg.norm(a)
    R = cp.zeros((mn, 1))

    if norm < tau:
        Q = None
    else:
        R[0, 0] = norm
        Q = 1 / norm * a
    for i in range(1, n):
        if Q is None:
            num_cols = 0
        else:
            num_cols = Q.shape[1]
        a = matrix[:, i][:, cp.newaxis]
        q = a
        r = fd(cp.zeros((mn, 1)))
        if num_cols < m:
            if num_cols > 0:
                q = q - fd(Q[:, :num_cols]).dot((a.T.dot(fd(Q[:, :num_cols]))).T)
                r[:num_cols, :] = fd(Q[:, :num_cols]).T.dot(a)
            norm = cp.linalg.norm(q)
            if norm < tau:
                if n - i < m - num_cols:
                    q = fd(cp.zeros((m, 1)))
            else:
                r[num_cols, 0] = 1
                r = r * norm
                q = q / norm
            if Q is None:
                Q = q
            else:
                Q = cp.hstack((Q, q))
        elif num_cols >= m:
            r[:m, :] = fd(Q[:, :num_cols]).T.dot(a)
        R = cp.hstack((R, r))
    return Q, R


def reduced_row_echelon_form_algorithm(matrix: cp.ndarray, tol: float = 1e-14):
    """Brings matrix into Reduced Row Echelon Form. Here, we implement the version from [2].

    References:
         [1] Heldt, D., Kreuzer, M., Pokutta, S. and Poulisse, H., 2009. Approximate computation of zero-dimensional
         polynomial ideals. Journal of Symbolic Computation, 44(11), pp.1566-1591.
         [2] Limbeck, J., 2013. Computation of approximate border bases and applications (Doctoral dissertation,
         Universität Passau).
    """
    matrix[cp.abs(matrix) < tol] = 0
    matrix = matrix[~cp.all(matrix == 0, axis=1)]
    matrix = row_sort(matrix)
    old_matrix = matrix
    while True:
        m, n = matrix.shape
        for i in range(0, m):
            if (matrix[i, :] != 0).any():
                j = int(cp.where(matrix[i, :][cp.newaxis, :] != 0)[1][0])
                tmp = cp.tile(matrix[i, j:], (matrix[:i, j:].shape[0], 1))
                vals = fd((matrix[:i, j] / tmp[:, 0]).flatten())
                matrix[:i, j:] = matrix[:i, j:] - cp.multiply(vals, tmp)
                if i < m - 1:
                    tmp = cp.tile(matrix[i, j:], (matrix[i + 1:, j:].shape[0], 1))
                    vals = fd((matrix[i + 1:, j] / tmp[:, 0]).flatten())
                    matrix[i + 1:, j:] = matrix[i + 1:, j:] - cp.multiply(vals, tmp)
        matrix[cp.abs(matrix) < tol] = 0
        matrix = matrix[~cp.all(matrix == 0, axis=1)]
        if old_matrix.shape == matrix.shape:
            if (cp.abs(old_matrix - matrix) <= tol).all():
                return row_sort(matrix)
        else:
            old_matrix = matrix


def find_range_null_vca(F: cp.ndarray, C: cp.ndarray, psi: float):
    """
    Performs FindRangeNull (using SVD) for VCA.

    Args:
        F: cp.ndarray
        C: cp.ndarray
        psi: float

    Returns:
        V_coefficient_vectors: cp.ndarray
            Coefficient vectors of the polynomials we append to V.
        V_evaluation_vectors: cp.ndarray
            Evaluation vectors of the polynomials we append to V.
        F_coefficient_vectors: cp.ndarray
            Coefficient vectors of the polynomials we append to F.
        F_evaluation_vectors: cp.ndarray
            Evaluation vectors of the polynomials we append to F.

    References:
        [1] Livni, R., Lehavi, D., Schein, S., Nachliely, H., Shalev-Shwartz, S. and Globerson, A., 2013, February.
        Vanishing component analysis. In International Conference on Machine Learning (pp. 597-605). PMLR.
    """
    F = fd(F)
    C = fd(C)
    tmp_coefficient_vectors = []
    tmp_evaluation_vectors = []
    for i in range(0, C.shape[1]):
        tmp_coefficient_vector = fd(cp.zeros(C.shape[1])).T
        tmp_coefficient_vector[:, i] = 1
        tmp_evaluation_vector = fd(C[:, i])
        orthogonal_components = orthogonal_projection(F, tmp_evaluation_vector)
        tmp_coefficient_vector = fd(cp.hstack((-orthogonal_components.T, tmp_coefficient_vector))).T

        tmp_evaluation_vector = tmp_evaluation_vector - fd(F.dot(orthogonal_components))
        assert (abs(
            fd(cp.hstack((F, C))).dot(tmp_coefficient_vector) - tmp_evaluation_vector) <= 10e-10).all(), "sanity check"
        tmp_coefficient_vectors.append(tmp_coefficient_vector)
        tmp_evaluation_vectors.append(tmp_evaluation_vector)

    tmp_coefficient_vectors = fd(cp.hstack(tmp_coefficient_vectors))
    tmp_evaluation_vectors = fd(cp.hstack(tmp_evaluation_vectors))
    A = tmp_evaluation_vectors
    if A.shape[0] > A.shape[1]:
        A_squared = A.T.dot(A)
        L, D, U = cp.linalg.svd(A_squared, full_matrices=True)
    else:
        L, D, U = cp.linalg.svd(A, full_matrices=True)

    U = fd(U)
    V_coefficient_vectors = []
    V_evaluation_vectors = []
    F_coefficient_vectors = []
    F_evaluation_vectors = []
    for i in range(0, C.shape[1]):
        row_U = fd(U[i, :]).T
        coefficient_vector = fd(tmp_coefficient_vectors.dot(row_U.T))
        evaluation_vector = fd(A.dot(row_U.T))

        loss = (1 / A.shape[0]) * cp.linalg.norm(evaluation_vector) ** 2

        if loss > psi:
            norm = cp.linalg.norm(evaluation_vector)
            F_coefficient_vectors.append(coefficient_vector / norm)
            F_evaluation_vectors.append(evaluation_vector / norm)
        else:
            V_coefficient_vectors.append(coefficient_vector)
            V_evaluation_vectors.append(evaluation_vector)

    if len(V_coefficient_vectors) > 0:
        V_coefficient_vectors = fd(cp.hstack(V_coefficient_vectors))
        V_evaluation_vectors = fd(cp.hstack(V_evaluation_vectors))
    if len(F_coefficient_vectors) > 0:
        F_coefficient_vectors = fd(cp.hstack(F_coefficient_vectors))
        F_evaluation_vectors = fd(cp.hstack(F_evaluation_vectors))

    return V_coefficient_vectors, V_evaluation_vectors, F_coefficient_vectors, F_evaluation_vectors
