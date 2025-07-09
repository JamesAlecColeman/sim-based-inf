import sys

running_on_arc = False

if running_on_arc:
    scripts_dir = "/home/scat8499/monoscription_python/JAC_Py_Scripts"
    sys.path.append(scripts_dir)

import twave_matching as twm
import ecg
import alg_utils
import os
from constants import *
import cache
from smoothing import preprocessing_gaussian_smoothing_fourier
import utils
import random
import time
import log_inference as log
import argparse
import shutil
from scipy.sparse.csgraph import dijkstra


def main():
    runtime_start = time.time()

    if running_on_arc:  # Remote ARC run
        parser = argparse.ArgumentParser()
        parser.add_argument('--benchmark_id', type=str, help='benchmark_id', required=True)
        parser.add_argument('--lambda_reg', type=str, help='lambda_reg', required=True)
        parser.add_argument('--seg_name', type=str, help='seg_name', required=True)
        parser.add_argument('--n_processors', type=str, help='n_processors', required=True)
        parser.add_argument('--n_tries', type=str, help='n_tries', required=True)
        parser.add_argument('--inferences_folder', type=str, help='inferences_folder', required=True)
        args = parser.parse_args()
        main_dir = "/data/coml-cardinal/scat8499/Monoscription"
        benchmark_id = args.benchmark_id
        seg_name = args.seg_name
        n_processors = int(args.n_processors)
        lambda_reg = float(args.lambda_reg)
        n_tries = int(args.n_tries)
        inferences_folder = args.inferences_folder
        patient_id = benchmark_id.split("_")[0]
        bench_dx = int(benchmark_id.split("_")[1])
        bench_type = benchmark_id.split("_")[2]
        save_best_every_x = 200

    else:  # Local run
        import addcopyfighandler
        main_dir = "C:/Users/jammanadmin/Documents/sim-based-inf-data"
        patient_id, bench_dx = "DTI024", 500
        bench_type = "hcmbig"
        benchmark_id = f"{patient_id}_{bench_dx}_{bench_type}"
        seg_name = "rvseg"
        n_tries = 512
        lambda_reg = 281.0
        n_processors = 4
        inferences_folder = "Inferences_twave_local"
        save_best_every_x = 1

    ############################################# Key Parameters #######################################################
    dx = 2000
    no_seg_dir = f"{main_dir}/no_segments"
    run_id = f"reg_{lambda_reg}_{seg_name}_{n_tries}_2daptable_moretrans"
    perform_logging = True
    n_iterations, percent_cutoff = 1000000, 87.5
    activation_start_s = 0.000
    iter_dt_activation_s, iter_dt_repol_s = 0.002, 0.010
    use_best_guess, use_monoalg_apd_field, segmental_monoalg_apds = 0, 0, 0
    use_clustered_output_params = 0
    plot, use_fibers, target_clinical = 0, 0, 0
    log_every_x_iterations = 1  # Must be every iteration to record all unique params, ECGs, RTs
    min_possible_apd90_ms, max_possible_apd90_ms, apd90_snapping_ms = 200, 400, 1
    mother_data_folder = "mother_data"
    ap_table_name = "ap_table_2d"
    fast_download_folder = f"fast_{benchmark_id}"
    use_best_inf_folder = "Inferences_no_segments"

    if use_best_guess:
        best_params_preload = np.load(f"{main_dir}/{use_best_inf_folder}/analysis/BESTPARAMS_{patient_id}_{bench_dx}_{bench_type}_reg_{lambda_reg}_{seg_name}_1024_2daptable_moretrans.npy", allow_pickle=True).item()

    mesh_dir = f"{main_dir}/Meshes_{dx}"

    # Load alg segmentation
    mesh_filename = utils.find_lvrv_thresh_used(mesh_dir, patient_id, dx, seg_name)
    alg_seg = alg_utils.read_alg_mesh(f"{mesh_dir}/{mesh_filename}")
    lv_rv = alg_seg[7]  # 0: rv, 1: lv
    trans = alg_seg[10]  # 0: endo, 1: epi
    apexb = alg_seg[14]
    n_cells = len(alg_seg[0])
    xs, ys, zs, *_ = alg_utils.unpack_alg_geometry(alg_seg)

    run_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{run_id}"
    print(f"{run_dir=}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    if not os.path.exists(f"{run_dir}/pop_ids_and_diffs"):
        os.makedirs(f"{run_dir}/pop_ids_and_diffs")

    # Optionally copies to run dir some best QRS params, activation times, target QRS, and target QRS+Twave
    if mother_data_folder is not None:
        mother_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{mother_data_folder}"
        shutil.copy(f"{mother_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams.npy",
                    f"{run_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams.npy")
        shutil.copy(f"{mother_dir}/{patient_id}_{dx}_activation_times.alg",
                    f"{run_dir}/{patient_id}_{dx}_activation_times.alg")
        shutil.copy(f"{mother_dir}/leads_selected_qrs.npy", f"{run_dir}/leads_selected_qrs.npy")
        shutil.copy(f"{mother_dir}/leads_selected_qrsandtwave.npy", f"{run_dir}/leads_selected_qrsandtwave.npy")

    # Prepare dijkstra distances for radius-wise apd manipulation
    dijk_dist_path = f"{no_seg_dir}/{patient_id}_{dx}_dijk_dists.npy"
    print("Prepare dijkstra distances")
    if os.path.exists(dijk_dist_path):
        # Load dijkstra dists if already saved
        all_dijk_dists_cm = np.load(dijk_dist_path)
    else:
        # Prepare dijkstra neighbourhoods for all points
        grid_dict = alg_utils.make_grid_dictionary(xs, ys, zs)
        adjacency_list_26 = ecg.compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, NEIGHBOURS_26)
        adjacency_matrix = twm.create_sparse_adjacency_distance(adjacency_list_26)
        all_dijk_dists_cm = dijkstra(adjacency_matrix, return_predecessors=False)  # Distances in cm
        np.save(dijk_dist_path, all_dijk_dists_cm)

    lead_names = LEAD_NAMES_12

    seg_field = alg_seg[-1]  # Works for RV seg

    seg_ids = np.unique(seg_field)

    if len(seg_ids) < 1 or len(seg_ids) > 500:
        raise Exception(
            f"{len(seg_ids)=} check alg_seg index for segmentation; have you accounted for the 6 geometry fields?")

    all_seg_idxs = [np.where(seg_field == seg_id)[0] for seg_id in seg_ids]  # Which mesh indices belong to each segment
    n_segments = len(seg_ids)
    print(f"{n_segments=}")

    # Get target leads into correct form
    leads_qrs = np.load(f"{run_dir}/leads_selected_qrs.npy", allow_pickle=True).item()
    leads_target = np.load(f"{run_dir}/leads_selected_qrsandtwave.npy", allow_pickle=True).item()
    times_target_s = leads_target[lead_names[0]][0]
    # TODO times target sanity check against total_time_s
    times_qrs_s = leads_qrs[lead_names[0]][0]
    leads_target_temp = {name: leads_target[name][1] for name in lead_names}
    leads_target = leads_target_temp
    target_qrsoff_s, target_end_time_s = max(times_qrs_s), max(times_target_s)

    smoothing_cutoff_s = target_qrsoff_s  # Smooth only during T wave
    activation_cutoff_s = smoothing_cutoff_s
    repol_start_s = smoothing_cutoff_s + iter_dt_activation_s
    total_time_s = target_end_time_s  # Target should have been padded during ECG subset selection

    print(f"Simulating up to {round(total_time_s * 1000)}ms")

    # Activation part
    times_activation_s = np.round(
        np.arange(activation_start_s, activation_cutoff_s + iter_dt_activation_s, iter_dt_activation_s), decimals=6)
    times_activation_s = times_activation_s[times_activation_s <= activation_cutoff_s]  # Prevent overstepping

    # Repolarisation part
    times_repol_s = np.round(np.arange(repol_start_s, total_time_s + iter_dt_repol_s, iter_dt_repol_s), decimals=6)
    times_repol_s = times_repol_s[times_repol_s <= total_time_s]  # Prevent overstepping

    times_s = np.concatenate((times_activation_s, times_repol_s))

    # Load AP table
    ap_table_2d = np.load(f"{main_dir}/{ap_table_name}.npy", allow_pickle=True).item()  # APD: [times_map_s, vms_new, mKr, mK1]
    ap_table_args = twm.preprocess_2d_ap_table(ap_table_2d, times_s, 5)

    (ap_table_arr, ap_table_rmps, min_apd90, max_apd90, min_apd50, max_apd50,
     apd90_step, apd50_step, ap_time_res_s, possible_apd50s_per_apd90) = ap_table_args

    # Loading geometry
    mesh_alg_name = f"{patient_id}_{dx}.alg"
    mesh_alg_path = f"{main_dir}/Meshes_{dx}/{mesh_alg_name}"
    print(f"Input alg mesh: {mesh_alg_path}")
    alg = alg_utils.read_alg_mesh(mesh_alg_path)
    xs, ys, zs, lxs, lys, lzs = alg_utils.unpack_alg_geometry(alg)
    dx = alg_utils.get_dx(xs)
    n_cells = len(xs)

    # Use of existing activation times from previous QRS personalisation
    qrsparams = np.load(f"{run_dir}/{patient_id}_{bench_dx}_ctrl_bestqrsparams.npy", allow_pickle=True)
    print(f"{qrsparams=}")
    v_myo_cm_per_s = qrsparams[0][1]
    conductivity = twm.monoalg_cv_to_conductivity(v_myo_cm_per_s)

    sigma_um_param = twm.monoalg_conductivity_to_smoothing_sigma(conductivity)


    print(f"{sigma_um_param=}")

    mesh_alg_activation_name = f"{patient_id}_{dx}_activation_times.alg"
    mesh_alg_activation_path = f"{run_dir}/{mesh_alg_activation_name}"
    alg_activation = alg_utils.read_alg_mesh(mesh_alg_activation_path)
    activation_times_s = np.array(alg_activation[-1])


    sigma_um = sigma_um_param
    manually_set_apd = None

    # Prepare dict to log inference parameters
    log_inf_params = {"main_dir": main_dir, "run_id": run_id, "patient_id": patient_id, "dx": dx,
                      "n_tries": n_tries, "n_iterations": n_iterations, "percent_cutoff": percent_cutoff,
                      "iter_dt_activation_s": iter_dt_activation_s, "iter_dt_repol_s": iter_dt_repol_s,
                      "use_fibers": use_fibers, "target_clinical": target_clinical,
                      "min_possible_apd90_ms": min_possible_apd90_ms,
                      "max_possible_apd90_ms": max_possible_apd90_ms, "apd90_snapping_ms": apd90_snapping_ms,
                      "sigma_um_param": sigma_um_param, "n_processors": n_processors,
                      "log_every_x_iterations": log_every_x_iterations, "n_segments": n_segments,
                      "qrsparams": qrsparams}

    # Read from cache
    cache_path = f"{main_dir}/Cache/{patient_id}_{dx}_cache.npy"
    mesh_info_dict = np.load(cache_path, allow_pickle=True).item()
    keys_to_read = ["endo_labels", "labels_meaning", "plane_6_params", "electrodes_xyz"]
    endo_labels, labels_meaning, plane_6_params, electrodes_xyz = cache.check_cache(mesh_info_dict, keys_to_read)

    # Preprocessing for pseudo ECG computation

    grid_dict = alg_utils.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, neighbour_arrays2 = ecg.get_neighbour_arrays(xs, ys, zs, dx, grid_dict)

    elec_grads = ecg.precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays).astype(np.float32)

    # Preprocess parts of smoothing
    x_i, y_i, z_i, vms_grid, dx, smoothed_mask = preprocessing_gaussian_smoothing_fourier(xs, ys, zs, sigma_um)

    possible_apd90s_ms = np.arange(min_possible_apd90_ms, max_possible_apd90_ms + 1, apd90_snapping_ms, dtype=np.int16)
    current_iter_params = {}
    params_included_already = set()

    n_attempts_init, max_attempts_init = 0, 1000
    ap_shape_param_init = 0.5  # 0.0: more triangular, 1.0 more rounded (APD50(APD90) function)

    for i_try in range(n_tries):

        while True:
            # Try adding uniform APDs among cells
            apd90_init_everywhere = random.choice(possible_apd90s_ms)
            apd90s_ms = np.ones(n_cells, dtype=np.int16) * apd90_init_everywhere
            params_dict_tup = {"apd90s_ms": tuple(apd90s_ms), "ap_shape_param": ap_shape_param_init}
            params_arr = {"apd90s_ms": apd90s_ms, "ap_shape_param": ap_shape_param_init}
            # dicts are not hashable so we must use the frozenset form to store them
            params_dict_frozen = frozenset(params_dict_tup.items())

            if params_dict_frozen not in params_included_already:
                # Add this APD param setup and proceed to next try
                current_iter_params[i_try] = params_arr
                params_included_already.add(params_dict_frozen)
                break
            else:
                # Perturb the configuration slightly to generate a unique one
                idx_perturb = np.random.randint(0, n_cells)

                # Select points in rad
                dijk_dists_cm = all_dijk_dists_cm[idx_perturb]
                idxs_in_rad = np.where(dijk_dists_cm <= 3.0)[0]

                apd90s_in_rad = apd90s_ms[idxs_in_rad].copy()

                new_apd90s_in_rad_temp = apd90s_in_rad * 1.05
                new_apd90s_in_rad = np.clip(new_apd90s_in_rad_temp, min_possible_apd90_ms, max_possible_apd90_ms)

                apd90s_ms[idxs_in_rad] = np.round(new_apd90s_in_rad).astype(np.int16)

                params_dict_tup = {"apd90s_ms": tuple(apd90s_ms), "ap_shape_param": ap_shape_param_init}
                params_arr = {"apd90s_ms": apd90s_ms, "ap_shape_param": ap_shape_param_init}
                # dicts are not hashable so we must use the frozenset form to store them
                params_dict_frozen = frozenset(params_dict_tup.items())

                if params_dict_frozen not in params_included_already:
                    # Add this perturbed APD param setup and proceed to next try
                    current_iter_params[i_try] = params_arr
                    params_included_already.add(params_dict_frozen)
                    break

            n_attempts_init += 1

            if n_attempts_init > max_attempts_init:
                raise Exception("Failed to initialise APDs with unique segmental APD distribution")

    if use_best_guess:
        n_tries = 1
        plot = 1
        n_iterations = 1
        current_iter_params = {0: best_params_preload}

    #  Ensure current_iter_params only contains dicts
    for key, value in current_iter_params.items():
        if isinstance(value, frozenset):
            current_iter_params[key] = dict(value)  # Convert frozenset to dict

    # Sanity check as number of unique initial APD distributions should equal n_tries
    if not use_best_guess and len(params_included_already) != n_tries:
        raise Exception(f"Initial number of unique APD parameter sets {len(params_included_already)} != {n_tries=}")

    activation_times_s = np.round(activation_times_s / ap_time_res_s) * ap_time_res_s

    print(f"{ap_time_res_s=}")

    if sigma_um <= 1000.0:
        print(f"Using low smoothing parameter {sigma_um=}, appropriate if setting based on ground truth APD field")

    # Pass in all the preprocessed arguments used for T wave computations
    repol_args_2daptable = (x_i.astype(np.int32), y_i.astype(np.int32), z_i.astype(np.int32), vms_grid.astype(np.int32), dx, smoothed_mask.astype(np.float32), sigma_um, smoothing_cutoff_s,
                           seg_ids, all_seg_idxs)

    count_x, count_y, count_z = neighbour_arrays["count_x"].astype(np.int32), neighbour_arrays["count_y"].astype(np.int32), neighbour_arrays["count_z"].astype(np.int32)
    valid_idxs, valid_positions = neighbour_arrays2["valid_idxs"].astype(np.int32), neighbour_arrays2["valid_positions"].astype(np.int32)
    valid_directions_for_neighbors = neighbour_arrays2["valid_directions_for_neighbors"].astype(np.int32)
    valid_offsets_for_neighbors = neighbour_arrays2["valid_offsets_for_neighbors"].astype(np.int32)

    neighbour_args = (
    count_x, count_y, count_z, valid_idxs, valid_positions, valid_directions_for_neighbors, valid_offsets_for_neighbors)

    alg = alg[:6]

    # QRS is now generated using the AP table to have a QRS of comparable amplitude to the T wave
    print("QRS part")
    all_activation_times_s = [activation_times_s]

    all_electrodes, *_ = twm.batch_ecg_runner(1, 1, twm.pseudo_ecg, times_activation_s,
                                            electrodes_xyz, elec_grads, dx, neighbour_args, all_all_vms=None,
                                             qrs_params=None,
                                             twave_params=current_iter_params,
                                             all_activation_times_s=all_activation_times_s,
                                             repol_args=repol_args_2daptable, calc_repol_times=False,
                                             ap_table_args=ap_table_args)

    if not use_best_guess and not use_clustered_output_params and perform_logging:
        log.log_init_twave(run_dir, log_inf_params, times_target_s, leads_target, alg, times_s, activation_times_s)

    leads_qrs_sim = ecg.ten_electrodes_to_twelve_leads(all_electrodes[0])

    mutated_params = {}  # Record all params tested and discrepancy
    all_ids_and_diff_scores = {}
    all_ids_and_grad_norms = {}

    runtimes = []

    compute_repolarisation_times = True

    # Main iterative refinement of T wave loop
    for iter_no in range(n_iterations):

        n_tries = len(current_iter_params)
        n_per_batch = int(round(n_tries / n_processors))
        n_per_batch = 1 if n_per_batch == 0 else n_per_batch
        tries = np.arange(n_tries)
        print(f"===================================== {iter_no} =====================================")
        print(f"n_tries = {n_tries}, n_per_batch = {n_per_batch}")

        all_activation_times_s = [activation_times_s for _ in range(n_tries)]  # All use same activation sequence

        # Compute electrode signals for these initial root nodes
        print(f"Number of parameter sets being ECG-tested: {len(current_iter_params)}")
        t0_batch = time.time()
        (all_electrodes, _, all_repol_times,
         all_vms_return,
         all_mean_mean_grad_norms) = twm.batch_ecg_runner(n_tries, n_per_batch, twm.pseudo_ecg, times_repol_s,
                                                         electrodes_xyz, elec_grads, dx, neighbour_args, all_all_vms=None,
                                                         qrs_params=None,
                                                         twave_params=current_iter_params,
                                                         all_activation_times_s=all_activation_times_s,
                                                         repol_args=repol_args_2daptable, all_apd_fields=manually_set_apd,
                                                         calc_repol_times=compute_repolarisation_times,
                                                         return_vms=False, ap_table_args=ap_table_args)
        t1_batch = time.time()
        print(f"{t1_batch - t0_batch} secs on batch ecgs")

        print(f"{round((t1_batch - t0_batch) / n_tries, 4)} secs elapsed per try")

        alg.append(activation_times_s)

        # Turn electrodes into leads and compare with the target leads
        population_diff_scores, all_leads_sim = {}, {}
        population_grad_norms = {}
        ids_and_ecgs_rts_params = {}  # To store repolarisation times and ECGs of this iteration
        population_reg_scores = {}
        population_ids = {}
        population_ids_check = {}

        for i_try in tries:
            leads_twave_sim = ecg.ten_electrodes_to_twelve_leads(all_electrodes[i_try])
            population_diff_scores[i_try], times_target_subset_s, all_leads_sim[
                i_try], target_full_ecg_leads_normed = twm.get_diff_score(times_repol_s, leads_twave_sim, times_target_s,
                                                                      leads_target, leads_qrs_sim, times_activation_s)
            population_grad_norms[i_try] = all_mean_mean_grad_norms[i_try]
            population_reg_scores[i_try] = population_diff_scores[i_try] + lambda_reg * all_mean_mean_grad_norms[i_try]

            param_id = twm.hash_twave_param(current_iter_params[i_try])

            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        for i_try, twave_param in current_iter_params.items():  # Store all params, diff scores, times and leads

            store_repol_times_ms = None

            if compute_repolarisation_times:
                store_repol_times_ms = np.round(np.array(all_repol_times[i_try]) * 1000)  # s to ms
                store_repol_times_ms = np.array(store_repol_times_ms, dtype=np.uint16)

            param_id = twm.hash_twave_param(twave_param)
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no, population_reg_scores[i_try]]

            ids_and_ecgs_rts_params[param_id] = [all_leads_sim[i_try], store_repol_times_ms, twave_param]
            all_ids_and_grad_norms[param_id] = all_mean_mean_grad_norms[i_try]

        # Retrieval of diff scores in the population but not simulated this iteration
        population_params = current_iter_params.copy()
        new_key = max(population_diff_scores.keys()) + 1


        for key, twave_param in mutated_params.items():  # mutated_params is of the population size
            # current_iter_params is of size n_tries of this iteration (unseen params)

            param_id = twm.hash_twave_param(twave_param)

            if param_id in all_ids_and_diff_scores and param_id not in population_ids_check:
                # Retrieving seen difference scores kept in the population but that were not simulated this iteration
                # Frozenset when used as key (to be hashable), dict when used as value (to be usable)
                population_diff_scores[new_key] = all_ids_and_diff_scores[param_id][0]
                population_reg_scores[new_key] = all_ids_and_diff_scores[param_id][2]
                population_grad_norms[new_key] = all_ids_and_grad_norms[param_id]
                population_params[new_key] = dict(twave_param)
                population_ids[new_key] = param_id
                population_ids_check[param_id] = 1
                new_key += 1

        print(f"Number of population diff scores: {len(population_diff_scores)}")

        # Find the keys with scores less than or equal to the n th percentile value
        scores = list(population_diff_scores.values())  # T wave discrepancy scores
        regularised_scores = list(population_reg_scores.values())

        percentile_thresh = np.percentile(regularised_scores, percent_cutoff)  # Uses regularised score
        keys_below = [key for key, value in population_reg_scores.items() if value <= percentile_thresh]  # Better

        keys_above = [key for key, value in population_reg_scores.items() if value > percentile_thresh]  # Worse

        min_diff_score = min(population_diff_scores.values())
        min_i_try_reg = min(population_reg_scores, key=population_reg_scores.get)
        best_reg_params = population_params[min_i_try_reg]
        hash_best_param = twm.hash_twave_param(best_reg_params)


        mutated_params = twm.mutate_twave_params_2daptable(keys_above, keys_below, population_params, possible_apd90s_ms,
                                                          min_possible_apd90_ms, max_possible_apd90_ms,
                                                          apd90_snapping_ms, all_dijk_dists_cm, trans, lv_rv, apexb)

        next_iter_params = {}
        next_iter_tries_ct = 0
        for key, twave_param in mutated_params.items():

            param_id = twm.hash_twave_param(twave_param)

            if param_id not in all_ids_and_diff_scores:  # Only simulate unseen params
                next_iter_params[next_iter_tries_ct] = twave_param
                next_iter_tries_ct += 1

        current_iter_params = next_iter_params

        if hash_best_param in ids_and_ecgs_rts_params:
            # Then the new best reg params were found this iteration (so update all best values)
            best_rts = ids_and_ecgs_rts_params[hash_best_param][1]
            best_apds = best_rts - activation_times_s*1000
            best_leads = ids_and_ecgs_rts_params[hash_best_param][0]

        best_apd90s_ms = best_reg_params["apd90s_ms"]
        best_apd50s_ms = np.empty(n_cells)

        best_ap_shape_param = best_reg_params["ap_shape_param"]

        for i, apd90 in enumerate(best_apd90s_ms):
            best_apd50s_ms[i] = twm.apd50_from_apd90(apd90, best_ap_shape_param, possible_apd50s_per_apd90[apd90])


        if iter_no % save_best_every_x == 0:  # Fast-download folder for at-a-glance inference evaluation

            alg = alg[:6]
            alg.append(best_apd90s_ms)
            alg.append(best_apd50s_ms)
            alg.append(best_rts)
            alg.append(best_apds)
            if use_best_guess:
                alg_utils.save_alg_mesh(f"{run_dir}/{fast_download_folder}/bestguess_best_reg_params_{iter_no}.alg", alg)
                np.save(f"{run_dir}/{fast_download_folder}/bestguess_best_leads_{iter_no}.npy", best_leads)
            else:
                alg_utils.save_alg_mesh(f"{run_dir}/{fast_download_folder}/best_reg_params_{iter_no}.alg", alg)
                np.save(f"{run_dir}/{fast_download_folder}/best_leads_{iter_no}.npy", best_leads)

        n_uniques = len(all_ids_and_diff_scores)

        mean_diff_score = np.mean(scores)

        if len(scores) >= 10:
            sorted_scores = sorted(scores)
            tenth_best_score = round(sorted_scores[9], 5)
        else:
            tenth_best_score = None

        #print("best reg params:", best_reg_params)
        print(f"Min: {round(min_diff_score, 5)}, 10th best: {tenth_best_score}, Mean: {round(mean_diff_score, 5)}, Uniques: {n_uniques}")


        if use_best_guess:
            print("All diff scores:", population_diff_scores.values())

        runtime_end = time.time()
        runtime_current_total = runtime_end - runtime_start
        runtimes.append(runtime_current_total)

        if not use_best_guess and not use_clustered_output_params and perform_logging:
            log.log_progress_twave_no_segments(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                                   population_ids, population_diff_scores, ids_and_ecgs_rts_params,
                                   population_reg_scores)
    if plot:
        # Simulated and target plot
        ecg.plot_ecg([times_s, times_target_s], [all_leads_sim[0], target_full_ecg_leads_normed], colors=["red", "black"], labels=["Inferred", "Target"])


if __name__ == '__main__':
    main()
