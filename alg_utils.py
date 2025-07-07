import numpy as np
import utils
import os


def read_alg_mesh(file_path):
    """ Loads alg file of form x, y, z, dx, dy, dz, fields, note geometry values all ints

    Args:
        file_path (string): path to .alg file to load

    Returns:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
    """
    with open(file_path, 'r') as alg_file:
        # Initialise alg list structure
        first_line = alg_file.readline()
        fields = [f.strip() for f in first_line.split(",")]
        #fields = first_line.split(",")
        alg = [[] for _ in range(len(fields))]

        # Process first line
        for i in range(len(alg)):
            alg[i].append(utils.safe_float(fields[i]))

        # Process the rest of the file
        for line in alg_file:
            fields = [f.strip() for f in line.split(",")]

            for i in range(len(alg)):
                alg[i].append(utils.safe_float(fields[i]))

    alg = [np.array(alg_entry) for alg_entry in alg]

    # Geometry x, y, z, dx, dy, dz stored as int64
    for i in range(6):
        alg[i] = alg[i].astype(np.int64)

    return alg


def unpack_alg_geometry(alg):
    """ Get mesh cell coords and dxs from the loaded alg

    Args:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]

    Returns:
        xs, ys, zs, dxs, dys, dzs (tuple of arrays)
    """
    return alg[0], alg[1], alg[2], alg[3], alg[4], alg[5]


def get_dx(xs):
    """ Compute dx from xs

    Args:
        xs (array): mesh coordinates along one axis

    Returns:
        dx (int): spatial discretisation along this axis
    """
    if not len(xs):
        raise Exception(f"{xs=} cannot calculate dx")
    dx = np.unique(np.diff(np.sort(np.unique(xs))))[0]
    return int(dx)


def alg_from_xs(xs, ys, zs, fields=None, dx=None):
    """ Convert xs, ys, zs to alg list

    Args:
        xs, ys, zs (arrays): mesh coordinates along the 3 axes
        fields (list of arrays): Optionally include fields in the alg
        dx (int): Optionally pre-specify dx or it will be computed

    Returns:
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
    """
    if dx is None:
        dx = get_dx(xs)
    lxs = np.array([dx / 2 for _ in range(len(xs))])
    alg = [xs, ys, zs, lxs, lxs, lxs]

    if fields is not None:
        for field in fields:
            alg.append(field)

    return alg


def make_grid_dictionary(xs, ys, zs, values=None):
    """ Store mesh coordinates as hash map

    Args:
        xs, ys, zs (arrays of floats): coordinates of mesh cell centres
        values (array): optional alternative to using indices as the value stored in dict

    Returns:
        grid_dict (dict): key (x, y, z) mapped onto original index (OR optionally values) of point in the mesh
    """

    if values is None:  # coord : original index
        grid_dict = {(x, y, z): idx for idx, (x, y, z) in enumerate(zip(xs, ys, zs))}
    else:  # coord : value
        grid_dict = {(x, y, z): val for x, y, z, val in (zip(xs, ys, zs, values))}

    return grid_dict


def save_alg_mesh(path, alg, remove_old=True):
    """ Save alg with any number of fields to a file.

    Args:
        path (str): Path to the file where the mesh data will be saved.
        alg (list of arrays): alg in format [xs, ys, zs, dxs, dys, zs, fields1, ...]
        remove_old (bool): Flag to indicate if an existing file should be removed before saving.
    """

    # TODO rounding of values to stop large .alg files

    # Ensure alg contains xs, ys, zs, dx, dx, dx
    if len(alg) < 6:
        raise Exception(f"Trying to save alg with only {len(alg)} fields")

    # Cast entries to numpy arrays with a warning
    for i, field in enumerate(alg):
        if not isinstance(field, np.ndarray):
            alg[i] = np.array(field)
            print(f"Alg field {i} casted to numpy array")

    # Create directory if it does not yet exist
    alg_dir = os.path.dirname(path)
    if not os.path.exists(alg_dir):
        os.makedirs(alg_dir)

    # Checking if file already exists
    utils.handle_preexisting_path(path, remove_old)

    field_lengths = []
    # Checks for boolean arrays in the alg and converts them to int to avoid saving True/False in the .alg
    for i, arr in enumerate(alg):
        field_lengths.append(len(arr))
        if arr.dtype == bool:
            alg[i] = arr.astype(int)

    if len(set(field_lengths)) != 1:
        raise ValueError(f"Uneven number of cells between alg fields! Check {field_lengths=}")

    n_cells = len(alg[0])
    n_fields = len(alg)

    # If no cells to save, don't try to save the mesh
    if n_cells <= 0:
        print("No cells to save")
        return -1

    # Determine the maximum column width for each field
    max_col_widths = [max(len(str(item)) for item in field) for field in alg]

    with open(path, 'a') as alg_file:
        for i in range(n_cells):
            # Build each line with appropriate formatting
            line_parts = []
            for p in range(n_fields):
                value = str(alg[p][i]) + ","
                padded_value = value.ljust(max_col_widths[p] + 2)
                line_parts.append(padded_value)
            line = ''.join(line_parts).rstrip()  # Remove any trailing spaces for the last column
            alg_file.write(line[:-1] + "\n")  # Remove final comma and add newline