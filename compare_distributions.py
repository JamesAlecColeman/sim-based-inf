import numpy as np
from scipy.stats import spearmanr

def abs_diffs(distr_a, distr_b):
    """Compute element-wise absolute differences and their mean.

    Args:
        distr_a (array-like): First array of values
        distr_b (array-like): Second array of values

    Returns:
        all_abs_diffs (ndarray): Element-wise absolute differences
        mean_diffs (float): Mean of the absolute differences
    """
    all_abs_diffs = np.abs(distr_a - distr_b)
    mean_diffs = np.mean(all_abs_diffs)
    return all_abs_diffs, mean_diffs


def correlation(distr_a, distr_b):
    """Compute Spearman correlation between two distributions.

    Args:
        distr_a (array-like): First array of values
        distr_b (array-like): Second array of values

    Returns:
        corr (float): Spearman correlation coefficient, or NaN if not computable.
    """
    if np.std(distr_a) == 0 or np.std(distr_b) == 0 or len(distr_a) == 0 or len(distr_b) == 0:
        return np.nan  # Handle cases where correlation cannot be computed
    corr, _ = spearmanr(distr_a, distr_b)
    return corr
