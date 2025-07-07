import numpy as np
import os


def safe_float(val):
    """Safely convert a string to a float.

    Args:
        val (str or any): Value to convert to float.

    Returns:
        float: Converted float value, or np.nan if conversion fails.
    """
    try:
        return float(val.strip())
    except (ValueError, TypeError, AttributeError):
        return np.nan


def handle_preexisting_path(path, remove_old):
    """Detect and optionally remove an existing file path.

    Args:
        path (str): The file path to check.
        remove_old (bool): Whether to remove the existing file if it exists.
    """
    if os.path.exists(path):
        if remove_old:
            os.remove(path)
        else:
            raise Exception("Alg file already exists. Set remove_old=True or delete it.")
    if os.path.exists(path):
        raise Exception("Alg file already exists, remove_old seems to have failed")


def calc_dist_sq(x_a, y_a, z_a, x_b, y_b, z_b):
    """Calculate squared Euclidean distance between two 3D points.

    Args:
        x_a, y_a, z_a (float): Coordinates of the first point.
        x_b, y_b, z_b (float): Coordinates of the second point.

    Returns:
        float: Squared Euclidean distance between the two points.
    """
    return (x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2


def find_files(directory, prefix):
    """Find files in a directory starting with a given prefix.

    Args:
        directory (str): Path to the directory to search in.
        prefix (str): Filename prefix to filter files.

    Returns:
        list of str: Filenames in the directory that start with the prefix.
    """
    return [file for file in os.listdir(directory) if file.startswith(prefix)]


def linear_interpolation(x1, y1, x2, y2, x):
    """Perform linear interpolation between two points.

    Args:
        x1, y1 (float): Coordinates of the first point.
        x2, y2 (float): Coordinates of the second point.
        x (float): x-value to interpolate at.

    Returns:
        float: Interpolated y-value corresponding to x.
    """
    m = (y2 - y1) / (x2 - x1)  # Slope
    return y1 + m * (x - x1)


def linear_interpolation_arrays(xs, ys, x):
    """Perform linear interpolation for arrays of points at a specific x value.

    Args:
        xs (list or array): x-coordinates of known points, must be sorted.
        ys (list or array): y-coordinates of known points, same length as xs.
        x (float): x-value to interpolate at, must be within the range of xs.

    Returns:
        float: Interpolated y-value corresponding to x.

    Raises:
        ValueError: If xs and ys lengths differ or x is out of bounds.
    """
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
    """Find the LV/RV threshold mesh filename used (LV/RV discerned with manual thresholding)

    Args:
        mesh_dir (str): Directory containing mesh files.
        patient_id (str): Identifier for the patient.
        dx (str): spatial discretisation
        seg_name (str): Name of the processed mesh type

    Returns:
        str: Filename of the matching `.alg` file.

    Raises:
        Exception: If zero or multiple matching files are found.
    """
    prefix = f"{patient_id}_{dx}_{seg_name}"
    filenames = [f for f in os.listdir(mesh_dir) if (f.startswith(prefix) and f.endswith(".alg"))]

    if len(filenames) == 1:
        return filenames[0]
    else:
        raise Exception(f"Filenames: {filenames} failed to find lvrv threshold in use in {mesh_dir}")