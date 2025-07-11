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
    """ Create a sparse adjacency matrix of distances between neighbouring mesh cells.

    Args:
        adjacency_list (dict): Dictionary mapping cell indices to lists of
                               (neighbour_index, displacement_vector) tuples.
                               Displacement is in micrometres (um).

    Returns:
        scipy.sparse.csr_matrix: Sparse matrix (n_cells x n_cells) with entries in centimetres (cm)
                                 representing Euclidean distances between connected neighbours.
    """
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
    """ Preprocess 2D AP table ready for use in repolarisation ECG fast simulations

    Args:
        ap_table_2d (dict): Dictionary mapping (APD90, APD50) tuples to tuples of (time_array, V_m_array).
                            Times are in seconds, Vm in mV.
        times_sim_s (np.ndarray): Array of simulation times in seconds.
        every_xth_time (int): Factor to reduce AP time resolution by (e.g. 5 means take every 5th time point).

    Returns:
        ap_table_arr (np.ndarray): 3D float32 array of shape (n_apd90s, n_apd50s, n_times_ap),
                                   holding the reduced AP voltage traces.
        ap_table_rmps (np.ndarray): 2D float32 array of resting membrane potentials for each APD90/APD50 combo.
        min_apd90 (float): Minimum APD90 value in the table.
        max_apd90 (float): Maximum APD90 value in the table.
        min_apd50 (float): Minimum APD50 value in the table.
        max_apd50 (float): Maximum APD50 value in the table.
        apd90_step (float): Step size between APD90 values (must be uniform).
        apd50_step (float): Step size between APD50 values (must be uniform).
        ap_time_res_s (float): Time resolution of the AP traces in seconds.
        possible_apd50s_per_apd90 (dict): Dictionary mapping each APD90 to the available APD50 values.
    """
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
    """ Convert conduction velocity (CV) in cm/s to MonoAlg model conductivity values
        based on a pre-tuned calibration curve.

        The calibration was performed using a 5 cm cable with 750 ms duration, where
        the end cell was stimulated at t=0 by -80 current, at a spatial resolution dx=500 um.

    Args:
        cv_cm_per_s (float or np.ndarray): Conduction velocity in centimeters per second.

    Returns:
        float or np.ndarray: Interpolated conductivity value(s) corresponding to the input CV(s).
    """
    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]
    cvs_cm_per_s = [0, np.float64(7.692307692307692), np.float64(15.527950310559007), np.float64(28.571428571428573), np.float64(39.37007874015748), np.float64(48.54368932038835), np.float64(56.81818181818182), np.float64(64.1025641025641), np.float64(70.42253521126761), np.float64(76.92307692307692), np.float64(81.9672131147541), np.float64(87.71929824561403), np.float64(92.5925925925926), np.float64(96.15384615384616)]
    conductivity_interp = utils.linear_interpolation_arrays(cvs_cm_per_s, conductivities, cv_cm_per_s)
    return conductivity_interp


def monoalg_conductivity_to_smoothing_sigma(conductivity, use_grads=True):
    """ Convert MonoAlg conductivity values to Gaussian smoothing parameter sigma.

        The smoothing parameter sigma was tuned using a 1 cm cube MonoAlg simulation (450 ms duration)
        with 70% endocardial and 30% epicardial cells, all stimulated at t=0 with a -53 current
        at dx=500 μm resolution.

        Two tuning methods are available:
        - Best raw Vm match (use_grads=False)
        - Best Vm gradient norm match (use_grads=True, default)

    Args:
        conductivity (float or np.ndarray): MonoAlg conductivity value(s).
        use_grads (bool): Whether to use gradient-based tuning (True) or raw Vm tuning (False).

    Returns:
        float or np.ndarray: Interpolated Gaussian smoothing sigma parameter(s).
    """

    conductivities = [0, 0.000025, 0.00005, 0.00010, 0.00015, 0.00020, 0.00025, 0.00030, 0.00035, 0.00040, 0.00045,
                      0.00050, 0.00055, 0.00060]

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
    """ Generate the membrane potential (Vm) field over time using a 2D AP table lookup and Gaussian smoothing

    Args:
        times_s (np.ndarray): Array of simulation time points in seconds.
        activation_times_s (np.ndarray): Activation times for each cell in seconds.
        twave_params (dict): Repolarisation parameters for the model
        ap_table_args (tuple): Tuple containing AP table arrays and metadata:
            (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
             apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90)
        repol_args_2daptable (tuple): Repolarisation arguments including grid and smoothing params:
            (x_i, y_i, z_i, vms_grid, dx, smoothed_mask, sigma_um, smoothing_cutoff_s,
             seg_ids, all_seg_idxs)
        dx (float): Spatial discretisation
        apd90_field_ms (np.ndarray, optional): Predefined APD90 values per cell in milliseconds.
        apd50_field_ms (np.ndarray, optional): Predefined APD50 values per cell in milliseconds.

    Returns:
        np.ndarray: 2D array (time x cells) of transmembrane voltages over time.
    """
    # Unpack AP table information
    (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
     apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90) = ap_table_args

    # Unpack repol_args
    x_i, y_i, z_i, vms_grid, dx, smoothed_mask, sigma_um, smoothing_cutoff_s = repol_args_2daptable

    if apd90_field_ms is not None and apd50_field_ms is not None:
        print("Using existing AP field")
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


def compute_batch_ecgs(pseudo_ecg_function, times_s, electrodes_xyz, elec_grads, dx,
                       neighbour_args, repol_args, batch_indices, batch_activation_times_s, batch_all_vms,
                       batch_twave_params, calc_repol_times=True, return_vms=False, ap_table_args=None):
    """ Compute pseudo ECG signals and repolarisation times for a batch of repol parameter sets

        For each trial in the batch:
        - Generate the Vm field
        - Optionally calculate repolarisation times from the Vm field.
        - Compute pseudo ECG signals and gradient norms (for regularisation) using the given pseudo_ecg_function.

    Args:
        pseudo_ecg_function (callable): Function that computes pseudo ECG and mean gradient norms.
        times_s (np.ndarray): Array of simulation time points in seconds.
        ap_function (callable): Action potential function.
        electrodes_xyz (np.ndarray): Electrode positions in 3D space.
        elec_grads (np.ndarray): Electrode gradient vectors.
        dx (float): Spatial discretisation
        activation_cutoff_s (float): Cutoff time for activation phase vs. repolarisation (controls smoothing)
        neighbour_arrays (list): Mesh structural info 1
        neighbour_arrays2 (list): Mesh structural info 2
        neighbour_args (dict): Additional arguments related to neighbours.
        repol_args (tuple or None): Arguments for repolarisation calculation and Vm smoothing.
        batch_indices (list): Indices of trials to process in this batch.
        batch_v_params (list): Parameters for each trial in the batch.
        batch_activation_times_s (list): Activation time arrays for each trial.
        batch_all_vms (list): Vm fields for each trial; will generate if None.
        batch_twave_params (list): T-wave parameters for each trial.
        calc_repol_times (bool, optional): Whether to calculate repolarisation times. Default is True.
        return_vms (bool, optional): Whether to return Vm fields for the batch. Default is False.
        ap_table_args (tuple, optional): AP table parameters needed for Vm generation.

    Returns:
        tuple:
            - dict: Computed pseudo ECG signals for each trial.
            - dict: Repolarisation times for each trial (if calculated).
            - list or None: Vm fields for each trial if return_vms=True, else None.
            - dict: Mean of mean gradient norms from pseudo ECG computation per trial.
    """
    batch_electrodes, batch_repol_times, batch_mean_mean_grad_norms = {}, {}, {}

    for i_try in batch_indices:
        # First generate Vms field based on the model parameters
        if batch_all_vms[i_try] is None and repol_args is not None:  # Only generate Vms field if you didn't input one

            vms_field = make_vms_field_2daptable(times_s, batch_activation_times_s[i_try],
                                                 batch_twave_params[i_try], ap_table_args,
                                                 repol_args, dx)
            batch_all_vms[i_try] = vms_field

            if calc_repol_times:  # Calculate post-smoothing repolarisation times
                n_cells = len(batch_activation_times_s[i_try])
                repol_times = np.empty(n_cells, dtype=float)
                none_repols = False

                for i_cell in range(n_cells):
                    apd, activation_time, repolarisation_time, ap_amp = moa.calc_apd_s(times_s, vms_field[:, i_cell])

                    if repolarisation_time is None:  # Record failed repolarisation in the mesh
                        none_repols = True
                        break

                    repol_times[i_cell] = repolarisation_time

                if none_repols:
                    repol_times = np.full(n_cells, 10.0, dtype=float)  # Sentinel value for failed repolarisation

                batch_repol_times[i_try] = repol_times

        # Calculate Pseudo ECG
        if batch_activation_times_s[i_try] is not None:
            activation = batch_activation_times_s[i_try]
        else:
            activation = None


        batch_electrodes[i_try], batch_mean_mean_grad_norms[i_try] = pseudo_ecg_function(times_s, electrodes_xyz,
                                                                                         elec_grads, dx, neighbour_args,
                                                                                         batch_all_vms[i_try])
    if return_vms:
        batch_vms_to_return = batch_all_vms
    else:
        batch_vms_to_return = None

    return batch_electrodes, batch_repol_times, batch_vms_to_return, batch_mean_mean_grad_norms


def batch_ecg_runner(n_tries, n_per_batch, pseudo_ecg_function, times_s, electrodes_xyz, elec_grads,
                     dx, neighbour_args, qrs_params=None,
                     all_activation_times_s=None, all_all_vms=None, all_apd_fields=None, twave_params=None,
                     repol_args=None, calc_repol_times=True, return_vms=False, ap_table_args=None):
    """ Run repolarisation ECG computations in parallel batches

    Args:
        n_tries (int): Total number of simulation tries.
        n_per_batch (int): Number of tries to process per batch.
        pseudo_ecg_function (callable): Function to compute pseudo ECG and gradients.
        times_s (np.ndarray): Simulation time points (seconds).
        ap_function (callable): Action potential function.
        electrodes_xyz (np.ndarray): 3D electrode coordinates.
        elec_grads (np.ndarray): Electrode gradient vectors.
        dx (float): Spatial discretisation
        activation_cutoff_s (float): When activation phase ends
        neighbour_arrays (list): Mesh structure info 1
        neighbour_arrays2 (list): Mesh structure info 2
        neighbour_args (dict): Additional neighbour-related arguments.
        qrs_params (list, optional): Parameters for QRS complexes per try.
        all_all_time_matrices (list, optional): Not currently used but reserved.
        all_activation_times_s (list, optional): Precomputed activation times per try.
        all_all_vms (list, optional): Precomputed Vm fields per try.
        all_apd_fields (list, optional): Predefined APD fields per try.
        twave_params (list, optional): T-wave parameters per try.
        repol_args (tuple, optional): Arguments controlling repolarisation smoothing and Vm generation.
        calc_repol_times (bool, optional): Whether to calculate repolarisation times. Default True.
        return_vms (bool, optional): Whether to return Vm fields after computation. Default False.
        ap_table_args (tuple, optional): AP table data

    Returns:
        tuple:
            - dict: Pseudo ECG electrode signals for each try.
            - dict: Recorded activation times per try (currently empty).
            - dict: Repolarisation times per try if calculated.
            - dict: Vm fields returned if requested; else empty.
            - dict: Mean of mean gradient norms per try from ECG computation.
    """
    all_electrodes, all_repol_times, all_vms_return, all_mean_mean_grad_norms = {}, {}, {}, {}
    batches = [range(i, min(i + n_per_batch, n_tries)) for i in range(0, n_tries, n_per_batch)]
    batched_v_params = [{} for _ in range(len(batches))]
    batched_activation_times_s = [{} for _ in range(len(batches))]
    batched_all_vms = [{} for _ in range(len(batches))]
    batched_twave_params = [{} for _ in range(len(batches))]
    batched_all_apd_fields = [{} for _ in range(len(batches))]

    record_activation_times_s = {}

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

    # Batch multiprocess parallel execution of activation times and pseudo ECG computation
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_batch_ecgs, pseudo_ecg_function, times_s, electrodes_xyz,
                                  elec_grads, dx, neighbour_args, repol_args, batch, batch_activation_times_s,
                                  batch_all_vms, batch_twave_params,
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

    return all_electrodes, record_activation_times_s, all_repol_times, all_vms_return, all_mean_mean_grad_norms

"""
def pseudo_ecg(times_s, ap_function, electrodes_xyz, elec_grads, dx,
                   activation_cutoff_s, neighbour_arrays, neighbour_arrays2, neighbour_args, v_params,
                   activation_times_s, all_vms, compute_grad_norms=True):
"""
def pseudo_ecg(times_s, electrodes_xyz, elec_grads, dx, neighbour_args, all_vms, compute_grad_norms=True):

    n_elec, n_cells, n_times = len(electrodes_xyz), elec_grads.shape[2], len(times_s)
    count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors, valid_offsets_for_neighbors = neighbour_args
    original_idxs = np.array(np.arange(0, n_cells, 1), dtype=int)

    if all_vms is None:
        raise Exception("Only works for precalculated Vms so far, otherwise vms = ap_function(time_point_s, activation_times_s)")

    electrodes_vs = np.zeros((n_elec, n_times))
    mean_grad_norms = []  # Store mean grad norm across cells for each time step

    for t_idx in range(n_times):

        grad = np.zeros((n_cells, 3))
        vms = all_vms[t_idx]
        vms_diff = vms[valid_idxs] - vms[valid_positions]  # Vm difference between neighbours (6 * n_cells,)

        # Adds to grad in vectorised fashion, but ensures multiple accesses don't get overwritten
        np.add.at(grad, (valid_positions, valid_directions_for_neighbors), (vms_diff / dx) * valid_offsets_for_neighbors)

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


def get_diff_score(times_sim_s, leads_twave_sim, times_target_s, leads_target, leads_qrs_sim,
                   times_activation_s, no_qrs=False):
    """ Calculate a T-wave difference score between simulated and target ECG leads, with QRS-based normalisation.

        Compares rescaled simulated T waves to target ECG data by aligning time points and scaling QRS amplitudes.
        The resulting score reflects average absolute differences across all 12 leads, scaled by target T wave amplitudes.

    Args:
        times_sim_s (np.ndarray): Time points for the simulated ECG (seconds).
        leads_twave_sim (dict): Simulated T-wave ECG signals (lead_name -> array).
        times_target_s (np.ndarray): Time points for the target ECG (seconds).
        leads_target (dict): Target ECG signals (lead_name -> array).
        leads_qrs_sim (dict): Simulated QRS ECG signals (lead_name -> array).
        times_activation_s (np.ndarray): Cell activation times (seconds), used to determine QRS window.
        no_qrs (bool, optional): If True, assumes no QRS data and scales using full ECG instead. Default False.

    Returns:
        tuple:
            - float or None: Mean normalised absolute difference across leads (None if no_qrs=True).
            - np.ndarray: Time points in the target ECG that match simulation times.
            - dict: Full rescaled simulated ECG (QRS + T) per lead.
            - dict: Target ECG signals at aligned time points per lead.
    """
    lead_names = LEAD_NAMES_12

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

    else:
        mean_sum_abs_diffs_normapproach = None

    return mean_sum_abs_diffs_normapproach, times_target_s[target_comparison_idxs], sim_full_ecg_leads, leads_target


def hash_twave_param(param):
    """ Hash a repolarisation model's params

    Args:
        param (dict): Dictionary of T-wave parameters (keys as strings, values can be scalars or numpy arrays).

    Returns:
        str: 8-character hexadecimal hash string representing the parameter set.
    """
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
    """ Replace worse-performing T-wave parameter sets with mutated versions of better-performing ones.

    Args:
        worse_keys (list): Keys corresponding to worse-performing parameter sets to be replaced.
        better_keys (list): Keys corresponding to better-performing parameter sets to sample from.
        all_params (dict): Dictionary mapping keys to T-wave parameter dictionaries.
        possible_apd90s_ms (list): List of valid APD90 values in ms.
        min_possible_apd90_ms (float): Minimum allowed APD90 value.
        max_possible_apd90_ms (float): Maximum allowed APD90 value.
        apd90_snapping_ms (float): Resolution step for snapping mutated APD90s.
        all_dijk_dists_cm (np.ndarray): Precomputed Dijkstra distance matrix in cm.
        trans (np.ndarray): Transmural coordinate per cell.
        lv_rv (np.ndarray): LV/RV coordinate per cell.
        apexb (np.ndarray): Apex-base coordinate per cell.

    Returns:
        dict: Updated `all_params` with mutated entries replacing the worse ones.
    """
    for i, worse_key in enumerate(worse_keys):

        replacement_apds = all_params[random.choice(better_keys)]
        mutated_replacement_apds = copy.deepcopy(replacement_apds)
        #  Prevents mutated parameters being identical to the originals
        while np.array_equal(mutated_replacement_apds["apd90s_ms"], replacement_apds["apd90s_ms"]):

            mutated_replacement_apds = mutate_twave_2daptable(replacement_apds, min_possible_apd90_ms, max_possible_apd90_ms,
                                                              apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb)
        all_params[worse_key] = mutated_replacement_apds

    return all_params


def mutate_twave_2daptable(replacement_params, min_possible_apd90_ms, max_possible_apd90_ms,
                           apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb):
    """ Applies a local mutation to a set of APD90 values and optionally mutates the AP shape parameter.

    Args:
        replacement_params (dict): Dict with keys 'apd90s_ms' and 'ap_shape_param' to be mutated.
        min_possible_apd90_ms (float): Minimum allowed APD90 after mutation.
        max_possible_apd90_ms (float): Maximum allowed APD90 after mutation.
        apd90_snapping_ms (float): Not used internally, kept for API compatibility.
        all_dijk_dists_cm (np.ndarray): Dijkstra distance array per point for defining spatial neighbourhoods.
        trans (np.ndarray): Array of transmural position values (0 = endo, 1 = epi).
        lv_rv (np.ndarray): Array of LV/RV labels per point (not used in this mutation).
        apexb (np.ndarray): Apex-base label or coordinates (not used in this mutation).

    Returns:
        dict: Mutated parameter dictionary with keys 'apd90s_ms' and 'ap_shape_param'.
    """
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

    min_apd_mult_explore, max_apd_mult_explore = 0.8, 1.2  # e.g. 300ms could go between 240-360ms
    min_apd_mult_exploit, max_apd_mult_exploit = 0.92, 1.08  # e.g. 300ms could go between 276-324ms

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
    """ Determines the APD50 value corresponding to a given APD90 using an AP shape parameter.

    Args:
        apd90 (float): The APD90 value in milliseconds.
        ap_shape_param (float): Value between 0 and 1 representing the relative steepness of the AP plateau.
        possible_apd50s (np.ndarray): Array of valid APD50 values for the given APD90.

    Returns:
        float: The selected APD50 value from `possible_apd50s`.
    """
    idx = round(ap_shape_param * (len(possible_apd50s) - 1))
    apd50 = possible_apd50s[idx]
    return apd50