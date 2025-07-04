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



def find_files(directory, prefix):
    return [file for file in os.listdir(directory) if file.startswith(prefix)]

def linear_interpolation(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)  # Slope
    return y1 + m * (x - x1)

def linear_interpolation_arrays(xs, ys, x):
    if len(xs) != len(ys):
        raise ValueError("Length of xs and ys must be the same.")
    if not (min(xs) <= x <= max(xs)):
        raise ValueError("x is out of bounds of the provided xs.")

    # Find the interval [x1, x2] that contains x
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            return linear_interpolation(xs[i], ys[i], xs[i + 1], ys[i + 1], x)

    raise ValueError("Failed interpolation")


def find_lvrv_thresh_used(mesh_dir, patient_id, dx, seg_name):

    prefix = f"{patient_id}_{dx}_{seg_name}"
    filenames = [f for f in os.listdir(mesh_dir) if (f.startswith(prefix) and f.endswith(".alg"))]

    if len(filenames) == 1:
        return filenames[0]
    else:
        raise Exception(f"Filenames: {filenames} failed to find lvrv threshold in use in {mesh_dir}")