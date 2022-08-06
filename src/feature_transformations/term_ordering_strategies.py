import cupy as cp
from src.auxiliary_functions.auxiliary_functions import fd
from src.auxiliary_functions.sorting import argsort_list


def pearson(data: cp.ndarray, rev: bool = False):
    """Determines an ordering of the columns according to the absolute values of the Pearson correlation coefficients.

    Example:
        Given
                A = [[1.         0.70923467 0.85606541]
                    [0.70923467 1.         0.24277391]
                    [0.85606541 0.24277391 1.        ]]
        we compute
                corr_gboeff = [[1.         0.70923467 0.85606541]
                            [0.70923467 1.         0.24277391]
                            [0.85606541 0.24277391 1.        ]]
        and
                corr_sum = [2.5653000783896567, 1.952008578360136, 2.0988393103629104].
        Then we call argsot_list() on corr_sum and if rev = True, reverse the result, that is, for rev = False, we get
        sort_gbols = [1, 2, 0] as the output of the function.

    Args:
        data: cp.ndarray
        rev: bool, Optional
            rev = False: The columns get sorted in increasing fashion.
            rev = True: The columns get sorted in decreasing fashion.
            (Default is False.)

    Returns:
        sort_gbols: list
            A list of integers such that data[:, sort_gbols] is such that its columns are increasing / decreasing in
            the sum of absolute correlation coefficients.
    """
    if fd(data).shape[1] == 1:
        return [0]
    else:
        corr_gboeff = cp.abs(cp.corrcoef(data, rowvar=False))
        corr_sum = cp.sum(corr_gboeff, axis=1).tolist()
        sort_gbols = argsort_list(corr_sum)
        if rev:
            sort_gbols.reverse()
        return sort_gbols
