from constants import *
import alg_utils
import numpy as np
import random
import hashlib
import math
import concurrent
import ecg
from scipy.sparse import csr_matrix
from itertools import islice
from scipy.sparse.csgraph import dijkstra


def mesh_subset_with_dist_constraint(alg, dist_limit_um, neighbour_dist_um):
    """ Deterministic sampling of alg cells where sampled points must be > dist_limit_um from all other sample
        points

    Args:
        alg (list): alg mesh
        dist_limit_um (float): mesh cells can only be sampled further than dist_limit from all other sampled cells
        neighbour_dist_um (float): neighbouring cells of sampled cells are stored as neighbours within this distance

    Returns:
        points_subset (list of float tuples): sampled points [(x0, y0, z0), (x1, y1, z1), ...]
        points_neighbours (dict): neighbours of points_subset stored as key (x, y, z): [(x0, y0, z0), ...]
    """
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg)
    grid_coarse = {}
    neighbours = np.concatenate((np.array([(0, 0, 0)]), NEIGHBOURS_26))  # Include same cell at origin
    # grid_scale_um must always be >= dist_limit_um to ensure dist limit is properly enforced
    grid_scale_um = dist_limit_um
    points_subset = []

    for x_cand, y_cand, z_cand in zip(xs, ys, zs):

        # Indices of the coarse grid which will contain the random point
        i_crs, j_crs, z_crs = int(x_cand // grid_scale_um), int(y_cand // grid_scale_um), int(z_cand // grid_scale_um)

        add_to_mesh = 1

        # Check neighbouring chunks for other sampled points
        for di, dj, dk in neighbours:
            ni, nj, nk = i_crs + di, j_crs + dj, z_crs + dk

            if (ni, nj, nk) in grid_coarse:
                # Check all points contained in this coarse grid entry
                for (x_other, y_other, z_other) in grid_coarse[(ni, nj, nk)]:
                    dist_sq = (x_cand - x_other) ** 2 + (y_cand - y_other) ** 2 + (z_cand - z_other) ** 2

                    # Refuse to add it to the mesh if it is too close to another point and stop checking this neighbour
                    if dist_sq <= dist_limit_um ** 2:
                        add_to_mesh = 0
                        break
            # If too close to another point, stop checking all neighbours as this candidate point is already rejected
            if not add_to_mesh:
                break

        # Add the successful candidate point to the coarse grid
        if add_to_mesh:

            # Initialise this coarse grid element if it does not yet exist
            if (i_crs, j_crs, z_crs) not in grid_coarse:
                grid_coarse[(i_crs, j_crs, z_crs)] = []

            grid_coarse[(i_crs, j_crs, z_crs)].append((x_cand, y_cand, z_cand))
            points_subset.append((x_cand, y_cand, z_cand))

    # New coarse grid suitable for distance checks up to neighbour_dist_um, formed from accepted the points subset
    grid_coarse = {}
    for x, y, z in points_subset:

        i_crs, j_crs, z_crs = int(x // neighbour_dist_um), int(y // neighbour_dist_um), int(z // neighbour_dist_um)

        # Initialise this coarse grid element if it does not yet exist
        if (i_crs, j_crs, z_crs) not in grid_coarse:
            grid_coarse[(i_crs, j_crs, z_crs)] = []

        # Add accepted point from points subset into the coarse grid
        grid_coarse[(i_crs, j_crs, z_crs)].append((x, y, z))

    # Neighbours of point (key (x, y, z)) stored as list of tuples [(x0, y0, z0), ...]
    points_neighbours = {}

    for x, y, z in points_subset:

        # Initialise neighbour list for this point
        points_neighbours[(x, y, z)] = []

        # Indices of the coarse grid which contains point (x, y, z)
        i_crs, j_crs, z_crs = int(x // neighbour_dist_um), int(y // neighbour_dist_um), int(z // neighbour_dist_um)

        # Check neighbouring chunks for other points
        for di, dj, dk in neighbours:
            ni, nj, nk = i_crs + di, j_crs + dj, z_crs + dk

            if (ni, nj, nk) in grid_coarse:
                # Check all points contained in this coarse grid entry
                for (x_other, y_other, z_other) in grid_coarse[(ni, nj, nk)]:
                    dist_sq = (x - x_other) ** 2 + (y - y_other) ** 2 + (z - z_other) ** 2

                    # (x_other, y_other, z_other) is classed as a neighbour if within this distance
                    if dist_sq <= neighbour_dist_um ** 2:
                        points_neighbours[(x, y, z)].append((x_other, y_other, z_other))

    return points_subset, points_neighbours


def init_roots_and_vels(n_tries, min_n_root_nodes, max_n_root_nodes, candidate_root_node_indices, v_endos_cm_per_s,
                        v_myos_cm_per_s, params_best_guess, use_best_guess):
    """ Initalises population of activation parameters

    Args:
        n_tries (int): population size
        min_n_root_nodes, max_n_root_nodes (int): min and max allowed number of root nodes of individual activation models
        candidate_root_node_indices (int list): idxs of allowed root node positions
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        params_best_guess (tuple tuple): for use when simulating just 1 activation model
        use_best_guess (bool): simulates just 1 activation model using params_best_guess if True

    Returns:
        current_iter_params (dict): population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
    """
    current_iter_params = {}

    if use_best_guess:  # Set the zeroth parameter combo to the best guess from before
        root_indices_mesh = list(params_best_guess[1])
        v_params = params_best_guess[0]
        current_iter_params[0] = (v_params, tuple(root_indices_mesh))
        return current_iter_params

    # Prepare initial root nodes
    for i_try in range(n_tries):
        n_root_nodes = random.randint(min_n_root_nodes, max_n_root_nodes)
        root_indices_mesh = []

        for _ in range(n_root_nodes):  # Select n random root nodes from candidate_root_node_indices
            rand_candidate_idx = random.randint(0, len(candidate_root_node_indices) - 1)
            root_indices_mesh.append(candidate_root_node_indices[rand_candidate_idx])

        # Select random v_endo_param, v_myo_param from the possible parameter sets
        v_endo_param, v_myo_param = random.choice(v_endos_cm_per_s), random.choice(v_myos_cm_per_s)
        v_params = (v_endo_param, v_myo_param)

        root_indices_mesh.sort()  # Prevent selection order mattering
        current_iter_params[i_try] = (v_params, tuple(root_indices_mesh))

    return current_iter_params


def create_sparse_adjacency_time(adjacency_list, v_fibers_cm_per_s, v_sheets_cm_per_s, v_normals_cm_per_s,
                                   v_endo_cm_per_s, endo_mask, use_fibers, fibers=None, sheets=None, normals=None):
    """ Creates sparse adjacency matrix representing travel times between cells using fast endo and myofibers

    Args:
        adjacency_list (dict): keys are cell indices and values are lists of tuples. Each tuple contains a neighboring
                               cell index and the displacement vector to that neighbor.
        v_fibers_cm_per_s, v_sheets_cm_per_s, v_normals_cm_per_s (floats): f, s, n conduction velocities
        v_endo_cm_per_s (float): isotropic endocardial conduction velocity
        endo_mask (bool array): flags endocardial cells in the alg mesh
        use_fibers (bool): uses fibers, sheets, normals vectors if True, defaults myo conduction to v_fibers if False
        fibers, sheets, normals (array of floats tuples): fiber, sheet, normal vectors for each cell

    Returns:
        Sparse adjacency matrix where each entry is of form (travel_time_ms, (idx, neighbor_idx))
    """
    row_indices, col_indices, data = [], [], []
    um_to_cm = 1e-4
    s_to_ms = 1000
    n_cells = len(adjacency_list)

    for idx, neighbors in adjacency_list.items():
        for neighbour_idx, displacement in neighbors:

            distance_um = np.linalg.norm(displacement)

            # Keep in mind 4 possibilities: endo-endo, endo-myo, myo-endo, myo-myo
            if endo_mask[idx] and endo_mask[neighbour_idx]:  # Isotropic fast endocardial propagation
                v_total_cm_per_s = v_endo_cm_per_s  # endo-endo

            elif use_fibers:  # Anisotropic bulk myocardial propagation using fibers
                disp_normed = displacement / distance_um
                f_vec, s_vec, n_vec = fibers[idx], sheets[idx], normals[idx]
                f_proj, s_proj, n_proj = np.dot(disp_normed, f_vec), np.dot(disp_normed, s_vec), np.dot(disp_normed, n_vec)
                v_f_proj, v_s_proj, v_n_proj = v_fibers_cm_per_s * np.abs(f_proj), v_sheets_cm_per_s * np.abs(s_proj), v_normals_cm_per_s * np.abs(n_proj)
                v_total_cm_per_s = np.sqrt(v_f_proj**2 + v_s_proj**2 + v_n_proj**2)

            else:  # Bulk myocardial propagation (no fibers, v_myo=v_fibers)
                v_total_cm_per_s = v_fibers_cm_per_s

            distance_cm = distance_um * um_to_cm
            travel_time_ms = distance_cm / v_total_cm_per_s * s_to_ms

            row_indices.append(idx)
            col_indices.append(neighbour_idx)
            data.append(travel_time_ms)

    return csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))


def hash_qrs_param(params):
    """ Hashes activation model params to give an identifier

    Args:
        params (tuple tuple): activation model params (v_endo_param, v_myo_param), (root_idx1, ...)

    Returns:
        string: 8 character hash identifying the activation model
    """
    param_str = str(params)
    hash_object = hashlib.md5(param_str.encode())
    return hash_object.hexdigest()[:8]


def mutate_activation_params(params, alg, grid_dict, candidate_root_node_indices, candidate_root_neighbours,
                             v_endos_cm_per_s, v_myos_cm_per_s, n_random_mutations=2, p_exploration=0.3,
                             p_velocity=0.3):
    """ Mutates activation parameters of a single activation model

    Args:
        params (tuple tuple): activation model params (v_endo_param, v_myo_param), (root_idx1, ...)
        alg (list): alg mesh
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        candidate_root_node_indices (int list): idxs of allowed root node positions
        candidate_root_neighbours (dict): neighbours of allowed root node posns stored as {(x, y, z): [(x0, y0, z0), ...], ...}
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        n_random_mutations (int): number of root node changes per mutation
        p_exploration (float): probability of explore-type mutation
        p_velocity (float): probability of mutating a conduction velocity param

   Returns:
       params (tuple tuple): mutated activation model params (v_endo_param, v_myo_param), (root_idx1, ...)
    """
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg)
    v_params, root_indices = params
    new_v_params, new_root_indices = v_params, list(root_indices).copy()

    if random.random() < p_velocity:  # Mutations applied to endo and myo conduction velocities
        v_endo_cm_per_s, v_myo_cm_per_s = v_params
        v_to_mutate = random.choice([0, 1])  # Randomly choose endo or myo velocity to mutate

        if v_to_mutate == 0:  # endo

            i_v_endo = v_endos_cm_per_s.index(v_endo_cm_per_s)
            v_endos_choice = []

            if i_v_endo != len(v_endos_cm_per_s) - 1:
                v_endos_choice.append(v_endos_cm_per_s[i_v_endo + 1])
            if i_v_endo != 0:
                v_endos_choice.append(v_endos_cm_per_s[i_v_endo - 1])

            if len(v_endos_cm_per_s) == 1:
                v_endos_choice = [v_endo_cm_per_s]

            new_v_endo_cm_per_s, new_v_myo_cm_per_s = random.choice(v_endos_choice), v_myo_cm_per_s

        elif v_to_mutate == 1: # myo
            i_v_myo = v_myos_cm_per_s.index(v_myo_cm_per_s)
            v_myos_choice = []

            if i_v_myo != len(v_myos_cm_per_s) - 1:
                v_myos_choice.append(v_myos_cm_per_s[i_v_myo + 1])
            if i_v_myo != 0:
                v_myos_choice.append(v_myos_cm_per_s[i_v_myo - 1])

            if len(v_myos_cm_per_s) == 1:
                v_myos_choice = [v_myo_cm_per_s]

            new_v_endo_cm_per_s, new_v_myo_cm_per_s = v_endo_cm_per_s, random.choice(v_myos_choice)

        new_v_params = (new_v_endo_cm_per_s, new_v_myo_cm_per_s)

    for _ in range(n_random_mutations):  # Mutations applied to root nodes
        # Root node to be replaced in the existing root_indices
        replace_node_idx = random.randint(0, len(new_root_indices) - 1)

        if random.random() < p_exploration:  # Exploration step (pick root node from all possible root node positions)
            rand_candidate_idx = random.randint(0, len(candidate_root_node_indices) - 1)
            new_root_indices[replace_node_idx] = candidate_root_node_indices[rand_candidate_idx]

        else:  # Exploitation step (mutate to a neighbouring root node)
            replace_mesh_idx = new_root_indices[replace_node_idx]
            x, y, z = xs[replace_mesh_idx], ys[replace_mesh_idx], zs[replace_mesh_idx]
            rand_neighbour = random.choice(candidate_root_neighbours[(x, y, z)])
            new_mesh_idx = grid_dict[rand_neighbour]
            new_root_indices[replace_node_idx] = new_mesh_idx

    new_root_indices.sort()
    params = new_v_params, tuple(new_root_indices)

    return params


def mutate_population_activation_params(worse_keys, better_keys, all_params, alg, grid_dict,
                                        candidate_root_node_indices, candidate_root_neighbours, v_endos_cm_per_s,
                                        v_myos_cm_per_s, all_ids_and_diff_scores):
    """ Applies replacement-mutation step to activation parameters of the activation model population

    Args:
        worse_keys (int list): [i_try, ...] corresponding to activation models with worse QRS match
        better_keys (int list): [i_try, ...] corresponding to activation models with better QRS match
        all_params (dict): population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
        alg (list): alg mesh
        grid_dict (dict): coordinate to mesh idx {(x, y, z): idx, ...}
        candidate_root_node_indices (int list): idxs of allowed root node positions
        candidate_root_neighbours (dict): neighbours of allowed root node posns stored as {(x, y, z): [(x0, y0, z0), ...], ...}
        v_endos_cm_per_s (float list): possible endocardial conduction velocities
        v_myos_cm_per_s (float list): possible myocardial conduction velocities
        all_ids_and_diff_scores (dict): records seen params {param_id: diff_score, iter_no}

    Returns:
        params_copy (dict): mutated population params {i_try: (v_endo_param, v_myo_param), (root_idx1, ...)}
    """

    params_copy = all_params.copy()

    # Replace worse models with random choice of better models (then mutate better models slightly)
    for i, worse_key in enumerate(worse_keys):
        replacement_params = params_copy[random.choice(better_keys)]  # Replace randomly with the better models
        n_mutation_attempts, max_mutation_attempts = 0, 1000

        while True:  # Keep mutating until you find a set of params not tested before
            mutated_replacement_params = mutate_activation_params(replacement_params, alg, grid_dict, candidate_root_node_indices,
                                                candidate_root_neighbours, v_endos_cm_per_s, v_myos_cm_per_s)

            if hash_qrs_param(mutated_replacement_params) not in all_ids_and_diff_scores:
                break  # Proceed with this mutated param as it is unseen

            if n_mutation_attempts > max_mutation_attempts:
                raise Exception("Failed to find a mutation that has not been tested before")
            n_mutation_attempts += 1

        params_copy[worse_key] = mutated_replacement_params

    return params_copy


def get_activation_times(root_indices, all_time_matrix):
    """Get activation times from time matrix using selected root node indices

    Args:
        root_indices (list of int): Indices of root nodes to use for activation
        all_time_matrix (ndarray): Time-to-arrival matrix (ms) for all candidate root nodes

    Returns:
        activation_times_s (ndarray): ventricular activation times using these root nodes
    """
    time_matrix = np.vstack([all_time_matrix[root_index] for root_index in root_indices])
    activation_times_s = np.min(time_matrix, axis=0) / 1000  # Convert milliseconds to seconds
    return activation_times_s


def pseudo_ecg_qrs(times_s, ap_function, electrodes_xyz, elec_grads, dx, activation_cutoff_s, neighbour_arrays,
                   v_params, activation_times_s):
    """ Pseudo ECG calculation for activation only

    Args:
        times_s (ndarray): 1D array of time points (s)
        ap_function (callable): Function that returns Vm array at a given time and activation map
        electrodes_xyz (ndarray): Coordinates of electrodes, shape (n_elec, 3)
        elec_grads (ndarray): Precomputed ∇(1/r) gradients from electrodes to each cell, shape (3, n_elec, n_cells)
        dx (float): Spatial discretisation
        activation_cutoff_s (float): Time until which activation-time-based gradient computation is allowed
        neighbour_arrays (dict): Mesh structural information precomputed
        v_params (list or None): Conduction velocity parameters (if provided, enables optimisation logic)
        activation_times_s (ndarray): Activation times for each cell (s)

    Returns:
        electrodes_vs (ndarray): Pseudo-ECG signals at each electrode over time, shape (n_elec, len(times_s))
    """
    n_elec = len(electrodes_xyz)

    if v_params is not None:
        using_activation_optimisation = True
        iter_dt_s = times_s[1] - times_s[0]
        min_v = min(v_params)
        min_dist_per_timestep_um = min_v * iter_dt_s * 10000
        n_time_steps_to_go_dx = math.ceil(dx / min_dist_per_timestep_um)
        first_activated_t_idxs = np.ceil(activation_times_s / iter_dt_s).astype(int)
    else:
        using_activation_optimisation = False

    unstructured_neighbour_idxs = neighbour_arrays["unstructured_neighbour_idxs"]

    electrodes_vs = np.zeros((n_elec, len(times_s)))

    # Compute ∇Vm at each time point
    for t_idx, time_point_s in enumerate(times_s):
        vms = ap_function(time_point_s, activation_times_s)  # Vms is the activated mask

        # Compute gradients considering which cells have actually been activated already
        if time_point_s <= activation_cutoff_s and using_activation_optimisation:

            t_idxs_to_consider = t_idx - np.arange(n_time_steps_to_go_dx)
            t_idxs_to_consider = t_idxs_to_consider[t_idxs_to_consider >= 0]
            activated_idxs_latest = np.where(np.isin(first_activated_t_idxs, t_idxs_to_consider))[0]

            neighbours_of_active_latest = unstructured_neighbour_idxs[activated_idxs_latest]
            valid_neighbors = neighbours_of_active_latest[neighbours_of_active_latest != -1]
            all_idxs = np.concatenate((activated_idxs_latest, valid_neighbors))
            calc_grad_at_idxs = np.unique(all_idxs)
            calc_grad_at_idxs = np.array(calc_grad_at_idxs, dtype=int)
            grad = ecg.calc_grads(np.array(vms), neighbour_arrays, dx, special_indices=calc_grad_at_idxs)
            grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]
            original_idxs = np.arange(0, len(grad), 1)

        else:  # Compute gradients for all cells
            grad = ecg.calc_grads(np.array(vms), neighbour_arrays, dx)
            grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]
            original_idxs = np.arange(0, len(grad), 1)

        original_idxs = np.array(original_idxs, dtype=int)

        # Dot ∇Vm with ∇(1/r)
        x_comp = grad_x[original_idxs].reshape(-1, 1) * elec_grads[0, :, original_idxs]
        y_comp = grad_y[original_idxs].reshape(-1, 1) * elec_grads[1, :, original_idxs]
        z_comp = grad_z[original_idxs].reshape(-1, 1) * elec_grads[2, :, original_idxs]

        # Sum the components along x, y, z for each electrode (sum over the n_cells dimension)
        electrodes_vs[:, t_idx] = -np.sum(x_comp + y_comp + z_comp, axis=0)

    return electrodes_vs


def action_potential_heaviside(t, activation_time):
    """ Computes the Heaviside step function at time t

    Args:
    t (float): present time
    activation_time (float): time cell is activated

    Returns:
    1 if t >= activation_time, 0 otherwise
    """
    return np.where(t >= activation_time, 1, 0)


def compute_batch_ecgs_qrs(pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads, dx, activation_cutoff_s,
                       neighbour_arrays, batch_indices, batch_v_params, batch_activation_times_s):
    """ Compute QRS pseudo-ECGs for a single batch of activation models.

    Args:
        pseudo_ecg_function (function): Function to compute ECG (e.g. pseudo_ecg_qrs).
        times_s (np.ndarray): Time points in seconds.
        ap_function (function): Function describing Vm(t), usually just step function for activation
        electrodes_xyz (np.ndarray): Shape (n_electrodes, 3), electrode positions
        elec_grads (np.ndarray): Shape (3, n_electrodes, n_cells), precomputed ∇(1/r) from each electrode to cell
        dx (float): Cell spacing in microns
        activation_cutoff_s (float): Time cutoff for 'activation phase'
        neighbour_arrays (dict): Precomputed mesh structural info
        batch_indices (list[int]): Indices of the current batch being processed.
        batch_v_params (dict[int, tuple]): Conduction velocities for each i_try in batch
        batch_activation_times_s (dict[int, np.ndarray]): Activation times (s) for each i_try in batch

    Returns:
        dict[int, np.ndarray]: Keys are i_try, values are ECG arrays of shape (n_electrodes, n_timepoints).
    """
    batch_electrodes = {}

    for i_try in batch_indices:
        batch_electrodes[i_try] = pseudo_ecg_function(times_s, ap_function, electrodes_xyz, elec_grads, dx,
                                                      activation_cutoff_s, neighbour_arrays, batch_v_params[i_try],
                                                      batch_activation_times_s[i_try])
    return batch_electrodes


def batch_ecg_runner_qrs(n_tries, n_per_batch, pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads,
                     dx, activation_cutoff_s, neighbour_arrays, all_all_time_matrices, qrs_params=None,
                     all_activation_times_s=None):
    """ Computes pseudo-ECGs for a population of activation models by parallelising across many batches

    Args:
        n_tries (int): Total number of models to compute ECGs for.
        n_per_batch (int): Number of models per parallel batch.
        pseudo_ecg_function (function): Function to compute pseudo-ECG from Vms
        times_s (np.ndarray): Time points (in seconds).
        ap_function (function): Vm(t), typically step function for activation
        electrodes_xyz (np.ndarray): Shape (n_electrodes, 3), electrode spatial positions.
        elec_grads (np.ndarray): Shape (3, n_electrodes, n_cells), ∇(1/r) values.
        dx (float): Spatial resolution of the mesh in microns.
        activation_cutoff_s (float): Time cutoff for 'activation phase'
        neighbour_arrays (dict): Precomputed mesh structural info
        all_all_time_matrices (dict): Precomputed general activation times (across possible root nodes, v_params)
        qrs_params (dict[int, tuple], optional): Maps i_try to (v_params, root_indices) tuples.
        all_activation_times_s (dict[int, np.ndarray], optional): Precomputed activation times if available

    Returns:
        all_electrodes (dict[int, np.ndarray]): ECG signals, one per i_try, shape (n_electrodes, n_timepoints).
        record_activation_times_s (dict[int, np.ndarray]): Activation times (s) per i_try.
    """
    all_electrodes = {}
    batches = [range(i, min(i + n_per_batch, n_tries)) for i in range(0, n_tries, n_per_batch)]
    batched_v_params = [{} for _ in range(len(batches))]
    batched_activation_times_s = [{} for _ in range(len(batches))]

    record_activation_times_s = {}

    # Precompute all activation times rather than pass in all_time_matrix to each subprocess
    for i, batch in enumerate(batches):
        for i_try in batch:
            if all_activation_times_s is not None:  # Then just use input activation times
                batched_activation_times_s[i][i_try] = all_activation_times_s[i_try]
            else:
                batched_activation_times_s[i][i_try] = None

            if qrs_params is not None and all_all_time_matrices is not None:
                # Then calculate activation times based on v_params, root_indices and the time matrices
                v_params, root_indices = qrs_params[i_try][0], qrs_params[i_try][1]

                all_time_matrix = all_all_time_matrices[v_params]
                activation_times_s = get_activation_times(root_indices, all_time_matrix)

                batched_v_params[i][i_try] = v_params
                batched_activation_times_s[i][i_try] = activation_times_s

                record_activation_times_s[i_try] = activation_times_s

            else:
                batched_v_params[i][i_try] = None

            if qrs_params is not None:  # v_params still needed for activation optimisation in pseudo ECG
                v_params = qrs_params[i_try][0]
                batched_v_params[i][i_try] = v_params

    # Batch multiprocess parallel execution of activation times and pseudo ECG computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_batch_ecgs_qrs, pseudo_ecg_function, times_s, ap_function,
                                   electrodes_xyz, elec_grads, dx, activation_cutoff_s, neighbour_arrays, batch,
                                   batch_v_params, batch_activation_times_s)
                   for batch, batch_v_params, batch_activation_times_s in zip(batches, batched_v_params, batched_activation_times_s)]

        # Add electrode outputs to the storage dictionary
        for future in concurrent.futures.as_completed(futures):
            batch_electrodes = future.result()
            all_electrodes.update(batch_electrodes)

    return all_electrodes, record_activation_times_s


def analyse_pseudo_electrodes_qrs(all_electrodes, target_leads,
                                  compare_with_target=True, lead_names_to_compare=None):
    """ Rescales simulated QRS and compares to target QRS

    Args:
        all_electrodes (dict): {i_try: [LA, RA, LL, RL, V1, V2, V3, V4, V5, V6], ...}
        target_leads (dict): {lead_name: signal, ...}
        compare_with_target (bool): computes difference score to target if True

    Returns:
        lead_diffs (dict): {lead_name: mean per-sample difference between a and b}
    """
    all_normed_leads_pseudo, all_diff_scores, all_leads_sim = {}, {}, {}

    for key, electrodes in all_electrodes.items():
        # Processing of pseudo ECG
        leads_pseudo = ecg.ten_electrodes_to_twelve_leads(electrodes)
        all_leads_sim[key] = leads_pseudo

        # Normalise pseudo ECG to amplitude of 1
        amplitudes_pseudo = {key: max(val) - min(val) for key, val in leads_pseudo.items()}
        normed_leads_pseudo = {key: val / amplitudes_pseudo[key] for key, val in leads_pseudo.items()}

        if compare_with_target:  # Compare to target QRS
            similarity_metric = measure_similarity_qrs(target_leads, normed_leads_pseudo, lead_names_to_compare=lead_names_to_compare)
            n_leads_compared = len(similarity_metric)
            diff_score = sum(similarity_metric.values()) / n_leads_compared
            all_diff_scores[key] = round(diff_score, 5)

        all_normed_leads_pseudo[key] = normed_leads_pseudo

    return all_normed_leads_pseudo, all_diff_scores, all_leads_sim


def measure_similarity_qrs(leads_a, leads_b, lead_names_to_compare=None):
    """ Compares the QRS similarity of two sets of leads.

    Args:
        leads_a (dict): {lead_name: signal, ...}
        leads_b (dict): {lead_name: signal, ...}

    Returns:
        lead_diffs (dict): {lead_name: mean per-sample difference between a and b}
    """
    lead_diffs = {}

    for i, key in enumerate(leads_a.keys()):

        if key in leads_b:  # Only compare leads in both target and pseudo ECG

            lead_a, lead_b = leads_a[key], leads_b[key]
            size_a, size_b = np.max(lead_a) - np.min(lead_a), np.max(lead_b) - np.min(lead_b)

            if not np.isclose(size_a, size_b, atol=0, rtol=1e-4):
                raise Exception("Amplitudes of the leads being compared is different")

            diffs = np.abs(lead_a - lead_b)

            if lead_names_to_compare is not None:
                if key in lead_names_to_compare:  # To just compare specific leads
                    lead_diffs[key] = sum(diffs) / len(lead_a)
            else:
                lead_diffs[key] = sum(diffs) / len(lead_a)  # Mean diffs per sample

    return lead_diffs


def find_optimal_scaling(leads_a, leads_b):
    """Compute optimal scaling factor to best match leads_a to leads_b

    Args:
        leads_a (dict): Dictionary of lead signals to be scaled, each value is a 1D numpy array.
        leads_b (dict): Dictionary of reference lead signals to match against, same structure as leads_a.

    Returns:
        float: Optimal scalar alpha such that alpha * leads_a ≈ leads_b
    """
    leads_a_signal = np.concatenate([leads_a[lead] for lead in leads_a])
    leads_b_signal = np.concatenate([leads_b[lead] for lead in leads_b])
    numerator = np.sum(leads_a_signal * leads_b_signal)
    denominator = np.sum(leads_a_signal ** 2)
    alpha = numerator / denominator if denominator != 0 else 0
    return alpha


def compute_time_matrix_batch(batch_args):
    """Compute time matrices for a batch of mesh arguments.

    Args:
        batch_args (list of tuples): Each tuple contains:
            - v_endo (array-like): endo conduction velocity
            - v_myo (array-like): myo conduction velocity
            - adjacency_list_26 (list): Adjacency info for 26-neighbour connectivity.
            - endo_mask (array-like): endocardial mask
            - use_fibers (bool): whether to use fiber velocities (not supported)
            - candidate_root_node_indices (list): Mesh idxs of candidate root nodes.

    Returns:
        dict: Keys are tuples (v_endo, v_myo), values are dicts mapping mesh index to
              time arrays computed from candidate root nodes.
    """
    results = {}
    for args in batch_args:
        (v_endo, v_myo, adjacency_list_26, endo_mask, use_fibers, candidate_root_node_indices) = args
        v_fibers, v_sheets, v_normals = v_myo, v_myo, v_myo
        adj_matrix = create_sparse_adjacency_time(
            adjacency_list_26, v_fibers, v_sheets, v_normals, v_endo, endo_mask, use_fibers
        )
        all_time_matrix = dijkstra(adj_matrix, indices=candidate_root_node_indices, return_predecessors=False)
        times_all_candidate_root_nodes = {
            mesh_idx: all_time_matrix[i].astype(np.float32)
            for i, mesh_idx in enumerate(candidate_root_node_indices)
        }
        results[(v_endo, v_myo)] = times_all_candidate_root_nodes
    return results



def batcher(iterable, batch_size):
    """Yield successive batches from an iterable.

    Args:
        iterable (iterable): The data source to be split into batches.
        batch_size (int): Number of items per batch.

    Yields:
        list: Next batch of items from the iterable.
    """
    it = iter(iterable)
    while True:
        batch = list(islice(it, batch_size))
        if not batch:
            break
        yield batch
