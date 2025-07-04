from constants import *
import utils
from collections import defaultdict
import matplotlib.pyplot as plt


def ten_electrodes_to_twelve_leads(electrodes):
    """ Converts 10 electrode signals into a 12-lead ECG

        Args:
            electrodes (list of arrays of floats): 10 electrode signals in order [LA, RA, LL, RL, V1, V2, V3, V4, V5, V6]

        Returns:
            leads (dict): A dictionary containing the computed 12-lead ECG signals with keys:
                  "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
    """

    n_electrodes = len(electrodes)
    if n_electrodes != 10:
        raise Exception(f"Trying to compute a 12-lead ECG from {n_electrodes} electrodes")

    leads = {}
    vw = 1 / 3 * (electrodes[0] + electrodes[1] + electrodes[2])
    leads["I"] = electrodes[0] - electrodes[1]
    leads["II"] = electrodes[2] - electrodes[1]
    leads["III"] = electrodes[2] - electrodes[0]
    leads["aVR"] = 3 / 2 * (electrodes[1] - vw)
    leads["aVL"] = 3 / 2 * (electrodes[0] - vw)
    leads["aVF"] = 3 / 2 * (electrodes[2] - vw)
    leads["V1"] = electrodes[4] - vw
    leads["V2"] = electrodes[5] - vw
    leads["V3"] = electrodes[6] - vw
    leads["V4"] = electrodes[7] - vw
    leads["V5"] = electrodes[8] - vw
    leads["V6"] = electrodes[9] - vw

    return leads


def get_neighbour_arrays(xs, ys, zs, dx, grid_dict):
    """ Precomputes all hex mesh structural relationships to speed up gradient/pseudo ECG calculations

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        dx (float): mesh spatial resolution
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}

    Returns:
        neighbour_arrays, neighbour_arrays2 (dicts): precomputed neighbourhood info
    """
    n_cells = len(xs)
    neighbours = NEIGHBOURS_FACE * dx

    pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs = np.ones(n_cells, dtype=int) * -1, np.ones(n_cells,
                                                                                               dtype=int) * -1, np.ones(
        n_cells, dtype=int) * -1, np.ones(n_cells, dtype=int) * -1, np.ones(n_cells, dtype=int) * -1, np.ones(n_cells,
                                                                                                              dtype=int) * -1

    unstructured_neighbour_idxs = np.empty((n_cells, 6))

    for i, (x, y, z) in enumerate(zip(xs, ys, zs)):

        for p, (di, dj, dk) in enumerate(neighbours):
            n_x, n_y, n_z = x + di, y + dj, z + dk

            if (n_x, n_y, n_z) in grid_dict:
                # Record presence of the neighbour in correct array

                n_idx = grid_dict[(n_x, n_y, n_z)]
                unstructured_neighbour_idxs[i, p] = n_idx

                if di > 0:
                    pos_xs[i] = n_idx
                elif di < 0:
                    neg_xs[i] = n_idx
                elif dj > 0:
                    pos_ys[i] = n_idx
                elif dj < 0:
                    neg_ys[i] = n_idx
                elif dk > 0:
                    pos_zs[i] = n_idx
                elif dk < 0:
                    neg_zs[i] = n_idx

            else:
                unstructured_neighbour_idxs[i, p] = -1  # Sentinel value

    unstructured_neighbour_idxs = np.array(unstructured_neighbour_idxs, dtype=object)

    # Count valid neighbours for each axis (x, y, z) per cell
    count_x = ((pos_xs != -1).astype(int) + (neg_xs != -1).astype(int))  # Count valid neighbours for x-direction
    count_y = ((pos_ys != -1).astype(int) + (neg_ys != -1).astype(int))  # Similarly for y-direction
    count_z = ((pos_zs != -1).astype(int) + (neg_zs != -1).astype(int))  # Similarly for z-direction

    neighbour_arrays = {}
    neighbour_arrays["pos_xs"], neighbour_arrays["neg_xs"], neighbour_arrays["pos_ys"], neighbour_arrays["neg_ys"], \
    neighbour_arrays["pos_zs"], neighbour_arrays["neg_zs"] = pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs
    neighbour_arrays["count_x"], neighbour_arrays["count_y"], neighbour_arrays["count_z"] = count_x, count_y, count_z
    neighbour_arrays["unstructured_neighbour_idxs"] = unstructured_neighbour_idxs

    original_idxs = np.array(np.arange(0, n_cells, 1), dtype=int)

    axes_arr = np.array([0, 0, 1, 1, 2, 2])  # x, y or z axis
    idxs_arr = np.array([pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs])  # mesh indices of neighbours in 6 directions
    offsets_arr = np.array([1, -1, 1, -1, 1, -1])  # positive or negative direction indicator

    valid_mask = (idxs_arr != -1)  # Denotes where neighbours actually exist (6, n_cells)
    valid_mask_flat = valid_mask.flatten()  # (6 * n_cells,)

    # Extract valid positions from the mask (valid positions in the mesh)

    valid_neighbors_idx = np.where(valid_mask_flat)[0]  # Flattened valid neighbor indices (< 6 * n_cells,)
    valid_directions = np.repeat(axes_arr, n_cells)  # Repeating directions for each neighbor (6 neighbors per cell)
    valid_offsets = np.repeat(offsets_arr, n_cells)  # Repeating offsets for each neighbor (6 neighbors per cell)
    tiled_idxs = np.tile(original_idxs, 6)

    # Now we can extract the correct direction and offset for each valid neighbor
    valid_directions_for_neighbors = valid_directions[valid_neighbors_idx]  # Directions corresponding to valid neighbors
    valid_offsets_for_neighbors = valid_offsets[valid_neighbors_idx]  # Offsets corresponding to valid neighbors
    valid_positions = tiled_idxs[valid_neighbors_idx]
    valid_idxs = idxs_arr.flatten()[valid_mask_flat]

    neighbour_arrays2 = {}
    neighbour_arrays2["valid_idxs"] = valid_idxs
    neighbour_arrays2["valid_positions"] = valid_positions
    neighbour_arrays2["valid_directions_for_neighbors"] = valid_directions_for_neighbors
    neighbour_arrays2["valid_offsets_for_neighbors"] = valid_offsets_for_neighbors

    return neighbour_arrays, neighbour_arrays2


def precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays):
    """ Precompute ∇(1/r) term for all electrodes for ECG calculation

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        electrodes_xyz (tuple list): electrode positions [(x1, y1, z1), ...]
        dx (float): mesh spatial resolution
        neighbour_arrays (dict): precomputed neighbourhood info

    Returns:
        elec_grads (float array): ∇(1/r) term of shape (3, n_elec, n_cells)
    """
    n_cells, n_elec = len(xs), len(electrodes_xyz)

    elec_grad_xs = np.empty((n_elec, n_cells), dtype=np.float64)
    elec_grad_ys = np.empty((n_elec, n_cells), dtype=np.float64)
    elec_grad_zs = np.empty((n_elec, n_cells), dtype=np.float64)

    for i, elec in enumerate(electrodes_xyz):
        rs_1over = 1 / np.sqrt(utils.calc_dist_sq(xs, ys, zs, elec[0], elec[1], elec[2]))
        elec_grads = calc_grads(rs_1over, neighbour_arrays, dx)
        elec_grad_xs[i, :], elec_grad_ys[i, :], elec_grad_zs[i, :] = elec_grads[:, 0], elec_grads[:, 1], elec_grads[:,
                                                                                                         2]
    elec_grads = np.stack([elec_grad_xs, elec_grad_ys, elec_grad_zs], axis=0)
    return elec_grads


def calc_grads(vms, neighbour_arrays, dx, special_indices=None):
    """ Computes gradients of field vms

    Args:
        vms (float array): scalar field to calculate gradients of
        neighbour_arrays (dict): precomputed neighbourhood info
        dx (float): mesh spatial resolution

    Returns:
        grad (float array): ∇vms of shape (n_cells, 3)
    """
    pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs = neighbour_arrays["pos_xs"], neighbour_arrays["neg_xs"], \
    neighbour_arrays["pos_ys"], neighbour_arrays["neg_ys"], neighbour_arrays["pos_zs"], neighbour_arrays["neg_zs"]
    count_x, count_y, count_z = neighbour_arrays["count_x"], neighbour_arrays["count_y"], neighbour_arrays["count_z"]

    n_cells = len(vms)

    # Initialize gradient array (n_cells x 3) to store gradients in x, y, and z directions
    grad = np.zeros((n_cells, 3))

    # Offsets for each direction (positive and negative)
    offsets = np.array([1, -1, 1, -1, 1, -1])  # For pos/neg x, y, z directions

    # Mask for special indices, if provided
    if special_indices is not None:
        special_mask = np.zeros(n_cells, dtype=bool)
        special_mask[special_indices] = True
    else:
        special_mask = None  # Process all cells

    # Compute gradients in the x-direction
    for direction, idxs, offset in zip(np.array([0, 0, 1, 1, 2, 2]),
                                       np.array([pos_xs, neg_xs, pos_ys, neg_ys, pos_zs, neg_zs]),
                                       np.array([0, 1, 2, 3, 4, 5])):

        # Mask out invalid neighbors
        valid_mask = idxs != -1

        if special_indices is not None:
            # Compute only for special indices
            valid_special_mask = valid_mask[special_indices]
            valid_special_indices = special_indices[valid_special_mask]
            grad[valid_special_indices, direction] += (vms[idxs[valid_special_indices]] - vms[valid_special_indices]) / dx * offsets[offset]
        else:
            # Compute for all cells
            grad[valid_mask, direction] += (vms[idxs[valid_mask]] - vms[valid_mask]) / dx * offsets[offset]

    # Avoid division by zero by checking the count for each cell
    grad[:, 0] /= np.maximum(count_x, 1)  # per-cell count for x direction
    grad[:, 1] /= np.maximum(count_y, 1)  # per-cell count for y direction
    grad[:, 2] /= np.maximum(count_z, 1)  # per-cell count for z direction

    return grad


def compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, neighbours):
    """ Compute adjacency list for mesh cells, providing displacement vectors (di, dj, dk) for neighbour cells

    Args:
        xs, ys, zs (float arrays): mesh cell center coordinates
        dx (float): mesh spatial resolution
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        neighbours (int array): local neighbourhood structure (see constants.py)

    Returns:
        adjacency_list (dict): of form key (cell index): [(neighbour index, (di, dj, dk)), ...]
    """
    # TODO: Use KD trees
    neighbours = neighbours * dx

    adjacency_list = defaultdict(list)

    for x, y, z in zip(xs, ys, zs):
        idx = grid_dict[(x, y, z)]

        for di, dj, dk in neighbours:  # Check each neighbour of the mesh cell to see if it exists
            ni, nj, nk = x + di, y + dj, z + dk

            if (ni, nj, nk) in grid_dict:  # Record neighbour idx and dist from point under original point idx as key
                neighbour_idx = grid_dict[(ni, nj, nk)]
                adjacency_list[idx].append((neighbour_idx, np.array([di, dj, dk])))

    if len(adjacency_list) != len(xs):
        raise Exception("Adjacency list not of same size as mesh")

    return adjacency_list


def get_closest_time(times_s, chosen_time_s):
    """ Finds index of element in times_s closest to chosen_time_s

        Args:
            times_s (float list): time axis points
            chosen_time_s (float): time you wish to find on time axis

        Returns:
            float: index of chosen_time_s in times_s
    """
    times_s = np.array(times_s)
    time_diffs_s = abs(times_s - chosen_time_s)
    return np.argmin(time_diffs_s)


def match_sim_and_target_times(times_sim_s, times_target_s):
    """ Finds indices of times_target_s which can be compared to times_sim_s

        Args:
            times_sim_s (float array): simulation time axis
            times_target_s (float array): target ECG time axis

        Returns:
            target_comparison_idxs (int array): indices of times_target_s corresponding to times_sim_s time points
    """
    n_sim_times = len(times_sim_s)

    target_comparison_idxs = np.empty(n_sim_times, dtype=int)

    for i, sim_time in enumerate(times_sim_s):
        target_comparison_idxs[i] = get_closest_time(times_target_s, sim_time)

    # Sanity check to ensure point on target ECG being compared to is close enough in time
    diff_tol_s = 0.001  # 1ms tolerance
    time_diffs_s = np.abs(times_target_s[target_comparison_idxs] - times_sim_s)
    max_time_diff_s = np.max(time_diffs_s)

    if max_time_diff_s > diff_tol_s:
        print(f"{times_sim_s=}")
        print(f"{times_target_s=}")
        print(f"{time_diffs_s=}")
        raise Exception(
            f"Matching time points with target ECG more than {diff_tol_s} secs apart, {max_time_diff_s}, can also be caused by the range of target data (end of target T wave less than max repol time you tried simulating)")

    return target_comparison_idxs


def plot_ecg(all_times_s, all_leads, colors=None, labels=None, linestyles=None, axes_off=True, xlims=None, show=True, fig_no=1,
             linewidth=1.5, show_zero=False, title=False, all_not_to_plot=None, text_overlays=None):
    plt.close(fig_no)

    n_ecgs = len(all_leads)

    if all_not_to_plot is None:
        all_not_to_plot = [[] for _ in range(n_ecgs)]

    if colors is None:
        colors = ["black" for _ in range(n_ecgs)]

    if labels is None:
        labels = [i for i in range(n_ecgs)]

    if linestyles is None:
        linestyles = ["-" for _ in range(n_ecgs)]

    # Plot
    lead_names = LEAD_NAMES_12

    width_px = 850
    height_px = 750
    dpi = 100  # TODO Should be 300 for publication
    width_in = width_px / dpi
    height_in = height_px / dpi

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(width_in, height_in), dpi=dpi, num=fig_no,
                             constrained_layout=True)

    if title is not False:
        fig.suptitle(title)

    axes = axes.flatten()
    for i, ax in enumerate(axes):

        lead_name = lead_names[i]
        ax.set_title(lead_names[i])
        ax.title.set_color('gray')

        if show_zero:
            ax.axhline(0, linestyle='--', color='grey')

        if axes_off:
            ax.axis('off')
            ax.title.set_bbox(dict(facecolor='none', edgecolor='none'))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.title.set_bbox(dict(facecolor='none', edgecolor='none'))

        if text_overlays is not None:
            overlay = text_overlays[i]
        else:
            overlay = ""

        ax.text(0.85, 0.9, overlay, transform=ax.transAxes, fontsize=8, color='blue', verticalalignment='top')

        for times_s, leads, color, label, linestyle, leads_not_to_plot in zip(all_times_s, all_leads, colors, labels, linestyles, all_not_to_plot):
            if lead_name in leads and lead_name not in leads_not_to_plot:
                ax.plot(times_s, leads[lead_name], color=color, label=label, linestyle=linestyle, linewidth=linewidth)



            if xlims is not None:
                ax.set_xlim(xlims)

    axes[0].legend(prop = {"size": 7})

    #plt.tight_layout()

    if show:
        plt.show()