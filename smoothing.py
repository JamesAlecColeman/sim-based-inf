from scipy.ndimage import gaussian_filter
import numpy as np


def preprocessing_gaussian_smoothing_fourier(xs, ys, zs, sigma_um):
    """ Prepare 3D grid from xs, ys, zs and apply Gaussian smoothing to the grid mask (mesh correction factor).

    Parameters:
        xs (array-like): x-coordinates of the points.
        ys (array-like): y-coordinates of the points.
        zs (array-like): z-coordinates of the points.
        sigma_um (float): Standard deviation of the Gaussian kernel in micrometers.

    Returns:
        x_i (np.ndarray): Integer indices of x-coordinates mapped to grid indices.
        y_i (np.ndarray): Integer indices of y-coordinates mapped to grid indices.
        z_i (np.ndarray): Integer indices of z-coordinates mapped to grid indices.
        vms_grid (np.ndarray): Zero-initialized grid for scalar field to be smoothed (membrane potentials)
        dx (float): Spatial discretisation
        smoothed_mask (np.ndarray): Gaussian-smoothed unit mask grid to be used as mesh smoothing correction factor
    """
    x_min, y_min, z_min = min(xs), min(ys), min(zs)
    x_max, y_max, z_max = max(xs), max(ys), max(zs)
    dx = np.unique(np.diff(np.sort(np.unique(xs))))[0]  # Cannot assume all xs will sequentially be separated by dx

    grid_shape = (int(np.round((x_max - x_min) / dx)) + 1,
                  int(np.round((y_max - y_min) / dx)) + 1,
                  int(np.round((z_max - z_min) / dx)) + 1)

    x_i = np.round((xs - x_min) / dx).astype(int)
    y_i = np.round((ys - y_min) / dx).astype(int)
    z_i = np.round((zs - z_min) / dx).astype(int)

    # Preprocess smoothing of the grid mask (which denotes which cells exist)
    mask_grid = np.zeros(grid_shape)
    mask_grid[x_i, y_i, z_i] = 1
    sigma_grid = sigma_um / dx
    smoothed_mask = gaussian_filter(mask_grid, sigma=sigma_grid)

    vms_grid = np.zeros(grid_shape)

    return x_i, y_i, z_i, vms_grid, dx, smoothed_mask


def gaussian_smoothing_fourier(vms, sigma_um, x_i, y_i, z_i, vms_grid, dx, smoothed_mask):
    """ Apply Gaussian smoothing to scalar field and correct for geometry bias.

    Parameters:
        vms (array-like): Values at mesh points to be smoothed
        sigma_um (float): Standard deviation of Gaussian kernel in micrometers.
        x_i, y_i, z_i (np.ndarray): Integer grid indices corresponding to the points.
        vms_grid (np.ndarray): 3D grid initialized to zeros for placing vms
        dx (float): Spatial discretisation
        smoothed_mask (np.ndarray): Mesh smoothing correction factor

    Returns:
        np.ndarray: Smoothed values
    """
    vms_grid[x_i, y_i, z_i] = vms
    sigma_grid = sigma_um / dx  # Sigma in um to grid spacing units
    smoothed_vms = gaussian_filter(vms_grid, sigma=sigma_grid)

    # Avoid division by zero in empty regions
    with np.errstate(divide='ignore', invalid='ignore'):
        masked_smoothed_vms = smoothed_vms / smoothed_mask  # Remove geometry bias from smoothing
        masked_smoothed_vms[smoothed_mask == 0] = 0  # Set empty regions back to 0

    return masked_smoothed_vms[x_i, y_i, z_i]