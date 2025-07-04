import numpy as np
from scipy.stats import spearmanr

def abs_diffs(distr_a, distr_b):
    all_abs_diffs = np.abs(distr_a - distr_b)
    mean_diffs = np.mean(all_abs_diffs)
    return all_abs_diffs, mean_diffs


def correlation(distr_a, distr_b):

    if np.std(distr_a) == 0 or np.std(distr_b) == 0 or len(distr_a) == 0 or len(distr_b) == 0:
        return np.nan  # Handle cases where correlation cannot be computed

    # Calculate Spearman's rank correlation
    corr, _ = spearmanr(distr_a, distr_b)

    return corr #np.corrcoef(distr_a, distr_b)[0, 1]
