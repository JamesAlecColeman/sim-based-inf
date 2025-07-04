import numpy as np
from scipy.sparse import csr_matrix
import utils
import hashlib
import monoalg_output_analysis as moa
import time
import concurrent
from smoothing import gaussian_smoothing_fourier
import copy
import random
from constants import *
import ecg

def create_sparse_adjacency_distance(adjacency_list):

    row_indices, col_indices, data = [], [], []
    um_to_cm = 1e-4
    n_cells = len(adjacency_list)

    for idx, neighbors in adjacency_list.items():
        for neighbour_idx, displacement in neighbors:

            distance_um = np.linalg.norm(displacement)
            distance_cm = distance_um * um_to_cm
            row_indices.append(idx)
            col_indices.append(neighbour_idx)
            data.append(distance_cm)
    return csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))


def preprocess_2d_ap_table(ap_table_2d, times_sim_s, every_xth_time):

    apd_tuples = np.array(list(ap_table_2d.keys()))
    max_sim_time = np.max(times_sim_s)

    for apd_tuple in apd_tuples:

        ts = ap_table_2d[tuple(apd_tuple)][0]
        vms = ap_table_2d[tuple(apd_tuple)][1]

        #  Reduce AP time resolution (0.2ms originally, typically we move to 1ms)
        ts_reduced = ts[::every_xth_time]
        vms_reduced = vms[::every_xth_time]

        up_to_sim_time_idxs = np.where(ts_reduced <= max_sim_time * 1.05)

        #  Reduce max AP time to the simulation time (originally 1000ms, typically goes down to about 500ms)
        ts_reduced = ts_reduced[up_to_sim_time_idxs]
        vms_reduced = vms_reduced[up_to_sim_time_idxs]

        ap_table_2d[tuple(apd_tuple)][0] = ts_reduced
        ap_table_2d[tuple(apd_tuple)][1] = vms_reduced

    ap_table_apd90s = apd_tuples[:, 0]
    ap_table_apd50s = apd_tuples[:, 1]

    min_apd90, max_apd90 = np.min(ap_table_apd90s), np.max(ap_table_apd90s)
    min_apd50, max_apd50 = np.min(ap_table_apd50s), np.max(ap_table_apd50s)

    unique_ap_table_apd90s = sorted(np.unique(ap_table_apd90s))
    d_apd90s = np.unique(np.diff(unique_ap_table_apd90s))

    if len(d_apd90s) > 1:  # Ensure AP table is equispaced in APD90
        raise Exception(f"AP table is not equispaced in APD90! Possible steps in APD90 in AP table are {d_apd90s}.")

    apd90_step = d_apd90s[0]

    possible_apd50s_per_apd90 = {}

    for apd90 in unique_ap_table_apd90s:
        # Finds all APD50s corresponding to this apd90
        requested_apd50s = apd_tuples[apd_tuples[:, 0] == apd90, 1]
        d_apd50s = np.unique(np.diff(requested_apd50s))
        possible_apd50s_per_apd90[apd90] = requested_apd50s

        if len(d_apd90s) > 1:  # Ensure AP table is equispaced in APD50
            raise Exception(f"AP table is not equispaced in APD50! Possible steps in APD90 in AP table are {d_apd50s}.")

    apd50_step = d_apd50s[0]

    times_ap = np.round(ap_table_2d[tuple(apd_tuples[0])][0], 8)  # Take any AP table times as they all share the time axis
    d_times_ap = np.unique(np.round(np.diff(times_ap), 8))  # Round to prevent float error

    if len(d_times_ap) > 1:  # Ensure AP table time axis is equispaced
        raise Exception(f"AP table time axis is not equispaced! Possible time steps are {d_times_ap}.")

    ap_time_res_s = d_times_ap[0]

    if not np.all(np.isin(times_sim_s, times_ap)):  # Ensure simulation times have a corresponding time in the AP table
        print(f"{times_sim_s=}")
        print(f"{times_ap=}")
        raise Exception("In the AP table time axis, there is not an entry corresponding to each simulation time point")

    n_times_ap = len(times_ap)
    n_apd90s = int((max_apd90 - min_apd90) / apd90_step) + 1
    n_apd50s = int((max_apd50 - min_apd50) / apd50_step) + 1

    # Every APD90 APD50 combo corresponding to a Vms array
    ap_table_arr = np.ones((n_apd90s, n_apd50s, n_times_ap)) * -1  # Sentinel value -1: no AP table entry
    ap_table_rmps = np.ones((n_apd90s, n_apd50s)) * -1  # Sentinel value -1: no AP table entry

    # Set up the final AP table array and AP RMP table array
    for apd_tuple in apd_tuples:
        apd90, apd50 = apd_tuple
        idx_apd90 = int((apd90 - min_apd90) / apd90_step)  # Zeroth column APD90 index
        idx_apd50 = int((apd50 - min_apd50) / apd50_step)  # First column APD50 index

        ap_table_arr[idx_apd90, idx_apd50] = ap_table_2d[tuple(apd_tuple)][1]
        ap_table_rmps[idx_apd90, idx_apd50] = ap_table_2d[tuple(apd_tuple)][1][-1]  # Last Vm value is taken as RMP

    ap_table_arr = ap_table_arr.astype(np.float32)
    ap_table_rmps = ap_table_rmps.astype(np.float32)

    return ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50, apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90


def monoalg_cv_to_conductivity(cv_cm_per_s):

    # Convert CV to a MonoAlg conductivity
    # Tuned based on 5cm cable 750ms duration, end cell stimulated at t=0 by -80 current, at dx=500um

    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]

    cvs_cm_per_s = [0, np.float64(7.692307692307692), np.float64(15.527950310559007), np.float64(28.571428571428573), np.float64(39.37007874015748), np.float64(48.54368932038835), np.float64(56.81818181818182), np.float64(64.1025641025641), np.float64(70.42253521126761), np.float64(76.92307692307692), np.float64(81.9672131147541), np.float64(87.71929824561403), np.float64(92.5925925925926), np.float64(96.15384615384616)]

    conductivity_interp = utils.linear_interpolation_arrays(cvs_cm_per_s, conductivities, cv_cm_per_s)
    return conductivity_interp


def monoalg_conductivity_to_smoothing_sigma(conductivity, use_grads=True):

    # Convert MonoAlg conductivity to a Gaussian smoothing parameter sigma
    # Smoothing parameter sigma tuned based on a 1cm cube MonoAlg simulation 450ms in duration with 70% endo, 30% epi cells
    # where all cells stimulated at t=0 with a -53 current, at dx=500um
    # Tuned in 2 possible ways: best raw Vm match, or best Vm gradient norm match
    # Can be refined using main_tune_smoothing.py

    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]

    # Based on dx=500um slab where AP table was the original one (based on stretching/compressing time axis)
    #best_sigmas_grad_vms = [0, 1250, 2350, 2850, 3200, 3550, 3900, 4200, 4550, 4800, 5000, 5200, 5400, 5550]
    #best_sigmas_vms      = [0, 1700, 2250, 3000, 3350, 3650, 4000, 4300, 4350, 4400, 4500, 4650, 4800, 4900]

    # Based on dx=500um slab where AP table used -known- pre-smoothing epi and endo AP shapes from MonoAlg
    best_sigmas_grad_vms = [np.float64(0.0), np.float64(1400.0), np.float64(1850.0), np.float64(2600.0),
                            np.float64(3050.0), np.float64(3250.0), np.float64(3500.0), np.float64(3650.0),
                            np.float64(3900.0), np.float64(4000.0), np.float64(4150.0), np.float64(4300.0),
                            np.float64(4450.0), np.float64(4550.0)]
    best_sigmas_vms = [np.float64(0.0), np.float64(1100.0), np.float64(1850.0), np.float64(2450.0), np.float64(2850.0),
                       np.float64(3200.0), np.float64(3500.0), np.float64(3800.0), np.float64(4050.0),
                       np.float64(4350.0), np.float64(4700.0), np.float64(4900.0), np.float64(5100.0),
                       np.float64(5250.0)]

    if use_grads:
        sigmas = best_sigmas_grad_vms
    else:
        sigmas = best_sigmas_vms

    sigma_interp = utils.linear_interpolation_arrays(conductivities, sigmas, conductivity)
    return sigma_interp

def make_vms_field_2daptable(times_s, activation_times_s, twave_params, ap_table_args, repol_args_2daptable, dx,
                             apd90_field_ms=None, apd50_field_ms=None):

    # Unpack AP table information
    (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
     apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90) = ap_table_args

    # Unpack repol_args
    (x_i, y_i, z_i, vms_grid, dx, smoothed_mask, sigma_um, smoothing_cutoff_s,
    seg_ids, all_seg_idxs) = repol_args_2daptable

    if apd90_field_ms is not None and apd50_field_ms is not None:
        print("Using existing AP field")  # TODO: add segmental handling
    else:

        ap_shape_param = twave_params["ap_shape_param"]

        # Construct the APD50 and APD90 field based on the parameters
        n_cells = len(activation_times_s)
        apd50_field_ms = np.empty(n_cells)
        apd90_field_ms = np.array(twave_params["apd90s_ms"])

        # Set cellwise APD50s from APD90s and the shape param
        for i, apd90 in enumerate(apd90_field_ms):
            apd50_field_ms[i] = apd50_from_apd90(apd90, ap_shape_param, possible_apd50s_per_apd90[apd90])


    if np.any(apd90_field_ms < 80) or np.any(apd50_field_ms < 80):
        raise ValueError("Small value in APD90/50 field: revisit how APD90/50 field is being set!")

    all_vms = np.zeros((len(times_s), len(activation_times_s)), dtype=float)  # Vms at time t

    idx_apd90_field = np.array((apd90_field_ms - min_apd90) / apd90_step, dtype=int)
    idx_apd50_field = np.array((apd50_field_ms - min_apd50) / apd50_step, dtype=int)

    # Initialise all_vms with RMPs corresponding to each APD
    for i, (idx_apd90, idx_apd50) in enumerate(zip(idx_apd90_field, idx_apd50_field)):
        all_vms[:, i] = ap_table_rmps[idx_apd90, idx_apd50]

    # Now construct Vm(t, x) based on which cells are already activated
    for i, t in enumerate(times_s):
        time_diffs = t - activation_times_s
        activated_mask = time_diffs >= 0

        # Compute the time index at each cell to access in the AP table
        time_idxs = ((time_diffs[activated_mask]) / ap_time_res_s).astype(int)
        all_vms[i][activated_mask] = ap_table_arr[
            idx_apd90_field[activated_mask], idx_apd50_field[activated_mask], time_idxs]

        # Smoothing Vms during repolarisation phase
        if t > smoothing_cutoff_s:  # Apply smoothing only during repolarisation (assumed steady state of diffusion)

            all_vms[i] = gaussian_smoothing_fourier(all_vms[i], sigma_um, x_i, y_i, z_i, vms_grid, dx, smoothed_mask)

    return all_vms


def compute_batch_ecgs(pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads, dx, activation_cutoff_s,
                       neighbour_arrays, neighbour_arrays2, neighbour_args, repol_args, batch_indices, batch_v_params, batch_activation_times_s, batch_all_vms,
                       batch_twave_params, batch_apd_fields, calc_repol_times=True, return_vms=False,
                       use_2daptable=False, ap_table_args=None, tied_segs=False):

    batch_electrodes, batch_repol_times, batch_mean_mean_grad_norms = {}, {}, {}

    #print("compute batch ecgs")

    for i_try in batch_indices:

        if batch_all_vms[i_try] is None and repol_args is not None:  # Only generate Vms field if you didn't input one

            vms_field = make_vms_field_2daptable(times_s, batch_activation_times_s[i_try],
                                                 batch_twave_params[i_try], ap_table_args,
                                                 repol_args, dx)

            batch_all_vms[i_try] = vms_field

            if calc_repol_times:  # Calculate post-smoothing repolarisation times
                n_cells = len(batch_activation_times_s[i_try])
                repol_times = np.empty(n_cells, dtype=float)
                none_repols = False

                for i_cell in range(n_cells):  # TODO explore the activation time input i_cell here and calc_apd function
                    apd, activation_time, repolarisation_time, ap_amp = moa.calc_apd_s(times_s, vms_field[:, i_cell])

                    if repolarisation_time is None:  # Record failed repolarisation in the mesh
                        none_repols = True
                        break

                    repol_times[i_cell] = repolarisation_time

                if none_repols:
                    repol_times = np.full(n_cells, 10.0, dtype=float)  # Sentinel value for failed repolarisation

                batch_repol_times[i_try] = repol_times

            else:
                print("NOT computing repol times")

        # TODO: as a debug step, we will save batch all vms generated by this step

        #main_dir = "C:/Users/jammanadmin/Documents/Monoscription"
        #np.save(f"{main_dir}/all_vms.npy", batch_all_vms)

        #neighbour_arrays, neighbour_arrays2 = None, None  # TODO remove if needed

        # Calculate Pseudo ECG

        if batch_activation_times_s[i_try] is not None:
            activation = batch_activation_times_s[i_try]
        else:
            activation = None

        batch_electrodes[i_try], batch_mean_mean_grad_norms[i_try] = pseudo_ecg_function(times_s, ap_function,
                                                                                         electrodes_xyz, elec_grads, dx,
                                                     activation_cutoff_s, neighbour_arrays, neighbour_arrays2, neighbour_args, batch_v_params[i_try],
                                                      activation, batch_all_vms[i_try])

    if return_vms:
        batch_vms_to_return = batch_all_vms
    else:
        batch_vms_to_return = None

    return batch_electrodes, batch_repol_times, batch_vms_to_return, batch_mean_mean_grad_norms

def batch_ecg_runner(n_tries, n_per_batch, pseudo_ecg_function, times_s, ap_function, electrodes_xyz, elec_grads,
                     dx, activation_cutoff_s, neighbour_arrays, neighbour_arrays2, neighbour_args, qrs_params=None, all_all_time_matrices=None,
                     all_activation_times_s=None, all_all_vms=None, all_apd_fields=None, return_activation_times=1, twave_params=None,
                     repol_args=None, calc_repol_times=True, return_vms=False, ap_table_args=None):

    # TODO: profile runtime of the batching overhead?
    t0 = time.time()

    all_electrodes, all_repol_times, all_vms_return, all_mean_mean_grad_norms = {}, {}, {}, {}
    batches = [range(i, min(i + n_per_batch, n_tries)) for i in range(0, n_tries, n_per_batch)]
    batched_v_params = [{} for _ in range(len(batches))]
    batched_activation_times_s = [{} for _ in range(len(batches))]
    batched_all_vms = [{} for _ in range(len(batches))]
    batched_twave_params = [{} for _ in range(len(batches))]
    batched_all_apd_fields = [{} for _ in range(len(batches))]

    record_activation_times_s = {}

    print("Into batch ECG runner")

    # Precompute all activation times rather than pass in all_time_matrix to each subprocess
    for i, batch in enumerate(batches):
        for i_try in batch:

            # main_twavematch QRS and T wave
            if all_activation_times_s is not None:
                batched_activation_times_s[i][i_try] = all_activation_times_s[i_try]
            else:
                batched_activation_times_s[i][i_try] = None

            if all_all_vms is None and repol_args is not None and twave_params is not None:
                # Then we need to set all_all_vms using twave_params and repol_args in each subprocess
                batched_twave_params[i][i_try] = twave_params[i_try]


            batched_v_params[i][i_try] = None

            if all_all_vms is not None:  # Manually set potentials
                # Vms at all times are already known so use these immediately
                batched_all_vms[i][i_try] = all_all_vms[i_try]
            else:
                batched_all_vms[i][i_try] = None

            if all_apd_fields is not None:  # We are setting all Vms based on a prespecified APD field
                batched_all_apd_fields[i][i_try] = all_apd_fields[i_try]
                print("Using a pre existing APD field")
            else:
                batched_all_apd_fields[i][i_try] = None

            if qrs_params is not None:  # v_params still needed for activation optimisation in pseudo ECG
                v_params = qrs_params[i_try][0]
                batched_v_params[i][i_try] = v_params

    exec_t0 = time.time()

    print("About to multiprocess parallel")
    # Batch multiprocess parallel execution of activation times and pseudo ECG computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # TODO consider using shared memory space because passing all structural data into workers is intensive
        futures = [executor.submit(compute_batch_ecgs, pseudo_ecg_function, times_s, ap_function, electrodes_xyz,
                                  elec_grads, dx, activation_cutoff_s, neighbour_arrays, neighbour_arrays2, neighbour_args, repol_args, batch,
                                   batch_v_params, batch_activation_times_s, batch_all_vms, batch_twave_params, batch_apd_fields,
                                   calc_repol_times=calc_repol_times, return_vms=return_vms,
                                   ap_table_args=ap_table_args)

                   for batch, batch_v_params, batch_activation_times_s, batch_all_vms, batch_twave_params, batch_apd_fields in zip(batches, batched_v_params,
                                                                                                                batched_activation_times_s,
                                                                                                                batched_all_vms, batched_twave_params, batched_all_apd_fields)]
        # Add electrode outputs to the storage dictionary
        for future in concurrent.futures.as_completed(futures):
            batch_electrodes, batch_repol_times, batch_vms_to_return, batch_mean_mean_grad_norms = future.result()
            all_electrodes.update(batch_electrodes)
            all_repol_times.update(batch_repol_times)
            all_mean_mean_grad_norms.update(batch_mean_mean_grad_norms)

            if batch_vms_to_return is not None:
                all_vms_return.update(batch_vms_to_return)

    exec_t1 = time.time()

    if not return_activation_times:
        batched_activation_times_s = None

    t1 = time.time()
    print(f"{round((t1 - t0) / n_tries, 4)} seconds elapsed per try")

    print(exec_t1 - exec_t0, "Parallel processing time")

    return all_electrodes, record_activation_times_s, all_repol_times, all_vms_return, all_mean_mean_grad_norms


def pseudo_ecg(times_s, ap_function, electrodes_xyz, elec_grads, dx,
                   activation_cutoff_s, neighbour_arrays, neighbour_arrays2, neighbour_args, v_params,
                   activation_times_s, all_vms, compute_grad_norms=True):

    n_elec, n_cells, n_times = len(electrodes_xyz), elec_grads.shape[2], len(times_s)
    count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors, valid_offsets_for_neighbors = neighbour_args
    original_idxs = np.array(np.arange(0, n_cells, 1), dtype=int)

    if all_vms is None:
        raise Exception("Only works for precalculated Vms so far, otherwise vms = ap_function(time_point_s, activation_times_s)")


    electrodes_vs = np.zeros((n_elec, n_times))

    # TODO custom np.add.at function that is numba compliant -and- faster, src: stackoverflow


    """# Version that is numpy vectorised over time axis too (more memory intensive)
    vms_diff_all = all_vms[:, valid_idxs] - all_vms[:, valid_positions]  # Shape: (n_times, 6 * n_cells)
    grad_all = np.zeros((n_times, n_cells, 3))
    np.add.at(grad_all, (np.arange(all_vms.shape[0])[:, None], valid_positions, valid_directions_for_neighbors),
              (vms_diff_all / dx) * valid_offsets_for_neighbors)

    # Normalize grad for each direction by the counts (which are the same across all time steps)
    grad_all[:, :, 0] /= np.maximum(count_x, 1)  # Normalize x direction
    grad_all[:, :, 1] /= np.maximum(count_y, 1)  # Normalize y direction
    grad_all[:, :, 2] /= np.maximum(count_z, 1)  # Normalize z direction

    # Extract the gradients for each direction
    grad_x_all, grad_y_all, grad_z_all = grad_all[:, :, 0], grad_all[:, :, 1], grad_all[:, :, 2]

    # Extract the gradient components for the selected cells and time points
    grad_x_all_selected = grad_x_all[:, original_idxs]  # Shape: (n_times, len(original_idxs))
    grad_y_all_selected = grad_y_all[:, original_idxs]  # Shape: (n_times, len(original_idxs))
    grad_z_all_selected = grad_z_all[:, original_idxs]  # Shape: (n_times, len(original_idxs))

    x_comp_all = grad_x_all_selected[:, :, None] * elec_grads[0, :, original_idxs]  # Shape: (n_times, len(original_idxs), n_electrodes)
    y_comp_all = grad_y_all_selected[:, :, None] * elec_grads[1, :, original_idxs]
    z_comp_all = grad_z_all_selected[:, :, None] * elec_grads[2, :, original_idxs]

    electrodes_vs = -np.sum(x_comp_all + y_comp_all + z_comp_all, axis=1)  # Shape: (n_electrodes, n_times)
    electrodes_vs = electrodes_vs.T"""

    mean_grad_norms = []  # Store mean grad norm across cells for each time step

    # For now, we completely ignore activation optimisation
    # Version without vectorisation over time axis (less memory intensive)
    for t_idx in range(n_times):

        grad = np.zeros((n_cells, 3))
        vms = all_vms[t_idx]

        vms_diff = vms[valid_idxs] - vms[valid_positions]  # Vm difference between neighbours (6 * n_cells,)

        # Adds to grad in vectorised fashion, but ensures multiple accesses don't get overwritten
        #grad[valid_positions, valid_directions_for_neighbors] += (vms_diff / dx) * valid_offsets_for_neighbors
        np.add.at(grad, (valid_positions, valid_directions_for_neighbors), (vms_diff / dx) * valid_offsets_for_neighbors)

        #add_at(grad, (valid_positions, valid_directions_for_neighbors), (vms_diff / dx) * valid_offsets_for_neighbors)

        # Avoid division by zero by checking the count for each cell
        grad[:, 0] /= np.maximum(count_x, 1)  # per-cell count for x direction
        grad[:, 1] /= np.maximum(count_y, 1)  # per-cell count for y direction
        grad[:, 2] /= np.maximum(count_z, 1)  # per-cell count for z direction

        grad_x, grad_y, grad_z = grad[:, 0], grad[:, 1], grad[:, 2]

        # Dot ∇Vm with ∇(1/r)
        x_comp = grad_x[original_idxs].reshape(-1, 1) * elec_grads[0, :, original_idxs]
        y_comp = grad_y[original_idxs].reshape(-1, 1) * elec_grads[1, :, original_idxs]
        z_comp = grad_z[original_idxs].reshape(-1, 1) * elec_grads[2, :, original_idxs]

        # Sum the components along x, y, z for each electrode (sum over the n_cells dimension)
        electrodes_vs[:, t_idx] = -np.sum(x_comp + y_comp + z_comp, axis=0)

        # Grad norm computation is for the regularisation term in the discrepancy metric
        if compute_grad_norms:
            grad_norms = np.linalg.norm(grad, axis=1)
            mean_grad_norm = np.mean(grad_norms)  # Mean across cells in single time step
            mean_grad_norms.append(mean_grad_norm)

    if compute_grad_norms:
        mean_mean_grad_norms = np.mean(mean_grad_norms)  # Mean across cells across all time steps
    else:
        mean_mean_grad_norms = None

    return electrodes_vs, mean_mean_grad_norms


def match_sim_and_target_times(times_sim_s, times_target_s):

    n_sim_times = len(times_sim_s)

    target_comparison_idxs = np.empty(n_sim_times, dtype=int)

    for i, sim_time in enumerate(times_sim_s):
        target_comparison_idxs[i] = get_closest_time(times_target_s, sim_time)

    # Sanity check to ensure point on target ECG being compared to is close enough in time
    diff_tol_s = 0.001  # 1ms tolerance
    time_diffs_s = np.abs(times_target_s[target_comparison_idxs] - times_sim_s)
    max_time_diff_s = np.max(time_diffs_s)
    i_max_time_diff = np.argmax(time_diffs_s)
    if max_time_diff_s > diff_tol_s:
        print(f"{times_sim_s=}")
        print(f"{times_target_s=}")
        print(f"{time_diffs_s=}")
        raise Exception(
            f"Matching time points with target ECG more than {diff_tol_s} secs apart, {max_time_diff_s}, can also be caused by the range of target data (end of target T wave less than max repol time you tried simulating)")

    return target_comparison_idxs


def get_diff_score(times_sim_s, leads_twave_sim, times_target_s, leads_target, leads_qrs_sim,
                   times_activation_s, no_qrs=False):
    lead_names = LEAD_NAMES_12

    # TODO: probably just consider 10 leads instead?, otherwise some electrodes overrepresented

    # Limit target ECG to just the QRS
    activation_cutoff_s = max(times_activation_s)
    target_activation_idxs = np.where(times_target_s <= activation_cutoff_s)[0]

    if no_qrs:  # No QRS: normalisation will be based on the T wave ranges
        sim_qrs_amps = {name: np.max(leads_twave_sim[name]) - np.min(leads_twave_sim[name]) for name in lead_names}
        target_activation_idxs = np.arange(len(times_target_s))
    else:  # QRS: normalisation will be based on QRS range
        sim_qrs_amps = {name: np.max(leads_qrs_sim[name]) - np.min(leads_qrs_sim[name]) for name in lead_names}

    # QRS amplitudes of target and simulated leads
    target_qrs_amps = {
        name: np.max(leads_target[name][target_activation_idxs]) - np.min(leads_target[name][target_activation_idxs])
        for name in lead_names}

    # Simply scales the simulated QRS to match the target QRS in amplitude
    lambda_scaling = {name: target_qrs_amps[name] / sim_qrs_amps[name] for name in lead_names}

    leads_qrs_sim_rescaled = {name: leads_qrs_sim[name] * lambda_scaling[name] for name in lead_names}
    leads_twave_sim_rescaled = {name: leads_twave_sim[name] * lambda_scaling[name] for name in lead_names}

    # Find indices of the target times to compare each simulation time to
    target_comparison_idxs = ecg.match_sim_and_target_times(times_sim_s, times_target_s)

    leads_twave_target = {name: leads_target[name][target_comparison_idxs] for name in lead_names}

    # For plotting
    sim_full_ecg_leads = {
        name: np.concatenate([np.array(leads_qrs_sim_rescaled[name]), np.array(leads_twave_sim_rescaled[name])]) for
        name in lead_names}

    # Each lead's T wave should be of amplitude 1 so that the diff score is fair among leads
    target_twave_amps = {name: np.max(leads_twave_target[name]) - np.min(leads_twave_target[name]) for
                         name in lead_names}

    if not no_qrs:
        # It's the absolute difference like before, but it's scaled to the target T wave amplitude
        abs_diffs_normapproach = {
            name: np.abs(leads_twave_sim_rescaled[name] - leads_twave_target[name]) / target_twave_amps[name] for name
            in lead_names}
        sum_abs_diffs_normapproach = {name: np.mean(abs_diffs_normapproach[name]) for name in lead_names}
        mean_sum_abs_diffs_normapproach = np.mean(list(sum_abs_diffs_normapproach.values()))

        """import ecg
        # Plot what is actually being compared
        ecg.plot_ecg([times_sim_s, times_sim_s], [leads_twave_sim_rescaled, leads_twave_target], show=True, colors=["red", "black"])

        # If computing correlation coefficient between T waves
        corrs = []
        for lead_name in LEAD_NAMES_12:
            corr = comp.correlation(leads_twave_sim_rescaled[lead_name], leads_twave_target[lead_name])
            corrs.append(corr)
        print(print(f"Corr of {np.mean(corrs)} +- {np.std(corrs)}"))"""

    else:
        mean_sum_abs_diffs_normapproach = None

    # TODO: consider doing each T wave peak individually. Maybe we can define a 'flat T wave' to avoid flat noisy waves coming up weirdly

    return mean_sum_abs_diffs_normapproach, times_target_s[target_comparison_idxs], sim_full_ecg_leads, leads_target


def hash_twave_param(param):
    h = hashlib.md5()
    # Iterate through dictionary and update hash
    for key, value in sorted(param.items()):
        h.update(key.encode('utf-8'))  #
        if isinstance(value, np.ndarray):
            h.update(value.tobytes())
        else:
            h.update(str(value).encode('utf-8'))
    # Return the first 8 characters of the hex
    return h.hexdigest()[:8]


def mutate_twave_params_2daptable(worse_keys, better_keys, all_params, possible_apd90s_ms, min_possible_apd90_ms,
                                  max_possible_apd90_ms, apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb):

    t0 = time.time()
    #params_copy = copy.deepcopy(all_params)  # Don't copy -every- parameter in there
    t1 = time.time()

    print(t1 - t0, "time on copying all params")

    t0 = time.time()

    # Replace worse models with random choice of better models, but mutated slightly
    for i, worse_key in enumerate(worse_keys):

        #replacement_apds = params_copy[random.choice(better_keys)]
        replacement_apds = all_params[random.choice(better_keys)]

        mutated_replacement_apds = copy.deepcopy(replacement_apds)

        a = 5

        #  Prevents mutated parameters being identical to the originals
        while np.array_equal(mutated_replacement_apds["apd90s_ms"], replacement_apds["apd90s_ms"]):

            mutated_replacement_apds = mutate_twave_2daptable(replacement_apds, min_possible_apd90_ms, max_possible_apd90_ms,
                                                              apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb)

        all_params[worse_key] = mutated_replacement_apds
        #params_copy[worse_key] = mutated_replacement_apds

    t1 = time.time()
    print(t1 - t0, "spent on the mutate_twave_2daptable")

    return all_params #params_copy

def mutate_twave_2daptable(replacement_params, min_possible_apd90_ms, max_possible_apd90_ms,
                           apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb):

    apd90s_ms = np.array(replacement_params["apd90s_ms"])
    apd90s_ms_new = apd90s_ms.copy()
    ap_shape_param = replacement_params["ap_shape_param"]

    shape_param_snapping = 0.05
    ap_shape_perturbation = np.arange(-0.20, 0.20 + shape_param_snapping, shape_param_snapping)
    ap_shape_perturbation = ap_shape_perturbation[~np.isclose(ap_shape_perturbation, 0.0)]

    p_change_ap_shapes = 0.2

    if random.random() <= p_change_ap_shapes:
        #  Mutate AP shape param
        possible_new_ap_shapes = ap_shape_perturbation + ap_shape_param
        # Restrict possible new ap shape params
        possible_new_ap_shapes = possible_new_ap_shapes[(possible_new_ap_shapes >= 0.0) & (possible_new_ap_shapes <= 1.0)]

        ap_shape_param_new = random.choice(possible_new_ap_shapes)
        #  Ensure it is to nearest 0.05
        ap_shape_param_new = np.round(ap_shape_param_new / shape_param_snapping) * shape_param_snapping
    else:
        ap_shape_param_new = ap_shape_param

    # TODO explore vs. exploit?

    # Perturb the APD90 field
    min_rad_cm, max_rad_cm = 1.0, 4.0
    min_poss_apd_ms, max_poss_apd_ms = 200, 400
    min_trans_amount, max_trans_amount = 0.3, 0.7
    p_endo_epi = 0.5

    p_transmurality = 0.25
    p_entire_ventricles_trans = 0.5

    p_explore = 0.3  # Chance of doing an explore mutation vs. exploit mutation
    n_apd_mutations_explore = 10
    n_apd_mutations_exploit = 3

    min_apd_mult_explore, max_apd_mult_explore = 0.8, 1.2
    min_apd_mult_exploit, max_apd_mult_exploit = 0.92, 1.08
    """p_mutate_seg_exploit = 0.1  # Average fraction of segments changed during exploit mutation
    apd90_perturbations_exploit_ms = np.arange(-30, 30 + apd90_snapping_ms, apd90_snapping_ms, dtype=np.int16)
    apd90_perturbations_exploit_ms = apd90_perturbations_exploit_ms[
        apd90_perturbations_exploit_ms != 0]  # Remove change of zero

    p_explore = 0.3  # Chance of doing an explore mutation vs. exploit mutation
    p_mutate_seg_explore = 0.3  # Average fraction of segments changed during explore mutation"""


    if random.random() <= p_explore:
        min_apd_mult, max_apd_mult = min_apd_mult_explore, max_apd_mult_explore  # Explore mutation
        n_apd_mutations = n_apd_mutations_explore
    else:
        min_apd_mult, max_apd_mult = min_apd_mult_exploit, max_apd_mult_exploit  # Exploit mutation
        n_apd_mutations = n_apd_mutations_exploit

    for _ in range(n_apd_mutations):
        idx = np.random.randint(0, len(apd90s_ms))
        rand_rad_cm = np.random.uniform(min_rad_cm, max_rad_cm)

        # Select points in rad
        dijk_dists_cm = all_dijk_dists_cm[idx]
        idxs_in_rad = np.where(dijk_dists_cm <= rand_rad_cm)[0]

        if random.random() <= p_transmurality:


            if random.random() <= p_entire_ventricles_trans:
                # Then just apply the transmural mutation to the entire ventricles, not just locally
                idxs_in_rad = np.where(dijk_dists_cm <= 20.0)[0]

            trans_in_rad = trans[idxs_in_rad]
            rand_trans_amount = np.random.uniform(min_trans_amount, max_trans_amount)

            if random.random() <= p_endo_epi:  # Mutation from endo side
                idxs_in_rad = idxs_in_rad[(trans_in_rad >= 0) & (trans_in_rad <= rand_trans_amount)]
            else:  # Mutation from epi side
                idxs_in_rad = idxs_in_rad[(trans_in_rad >= 1 - rand_trans_amount) & (trans_in_rad <= 1)]

        apd90s_in_rad = apd90s_ms_new[idxs_in_rad].copy()

        rand_apd_mult = np.random.uniform(min_apd_mult, max_apd_mult)
        new_apd90s_in_rad_temp = apd90s_in_rad * rand_apd_mult
        new_apd90s_in_rad = np.clip(new_apd90s_in_rad_temp, min_poss_apd_ms, max_poss_apd_ms)

        apd90s_ms_new[idxs_in_rad] = np.round(new_apd90s_in_rad).astype(np.int16)

    replacement_params_new = {"apd90s_ms": apd90s_ms_new.astype(np.int16), "ap_shape_param": ap_shape_param_new}

    return replacement_params_new


def apd50_from_apd90(apd90, ap_shape_param, possible_apd50s):
    """ Use AP shape param to find which APD50 corresponds to the set APD90 (for 2D AP table) """
    idx = round(ap_shape_param * (len(possible_apd50s) - 1))
    apd50 = possible_apd50s[idx]
    return apd50