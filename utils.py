import numpy as np
import os

def safe_float(val):
    """Safely convert a string to float. Returns np.nan if conversion fails."""
    try:
        return float(val.strip())
    except (ValueError, TypeError, AttributeError):
        return np.nan


def handle_preexisting_path(path, remove_old):
    """ Detects/removes existing path """
    if os.path.exists(path):
        if remove_old:
            os.remove(path)
        else:
            raise Exception("Alg file already exists. Set remove_old=True or delete it.")
    if os.path.exists(path):
        raise Exception("Alg file already exists, remove_old seems to have failed")


def calc_dist_sq(x_a, y_a, z_a, x_b, y_b, z_b):
    """ Get Euclidian distance squared between two points """
    return (x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2