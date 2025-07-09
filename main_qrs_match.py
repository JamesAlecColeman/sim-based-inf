import sys

running_on_arc = False

if running_on_arc:
    scripts_dir = "Your ARC Python directory here"
    sys.path.append(scripts_dir)

import time
import alg_utils
import os
import cache
from constants import *
import log_inference as log
import argparse
import qrs_matching as qrsm
import ecg
import math
import concurrent.futures


def main():
    runtime_start = time.time()
    if running_on_arc:  # Setup remote ARC run
        parser = argparse.ArgumentParser()
        parser.add_argument('--benchmark_id', type=str, help='benchmark_id', required=True)
        parser.add_argument('--n_processors', type=str, help='n_processors', required=True)
        parser.add_argument('--n_tries', type=str, help='n_tries', required=True)
        parser.add_argument('--inferences_folder', type=str, help='inferences_folder', required=True)
        args = parser.parse_args()
        main_dir = "/data/coml-cardinal/scat8499/Monoscription"
        benchmark_id = args.benchmark_id
        n_processors = int(args.n_processors)
        n_tries = int(args.n_tries)
        inferences_folder = args.inferences_folder
        patient_id = benchmark_id.split("_")[0]
        save_best_every_x = 200

    else:  # Setup local run
        import addcopyfighandler
        main_dir = "C:/Users/jammanadmin/Documents/sim-based-inf-data"
        patient_id, bench_dx = "DTI024", 500
        bench_type = "ctrl"
        benchmark_id = f"{patient_id}_{bench_dx}_{bench_type}"
        n_tries = 512
        n_processors = 4
        inferences_folder = "Inferences_qrs_local"
        save_best_every_x = 1

    ############################################# Key Parameters #######################################################
    run_id = f"sim-based-inf-{n_tries}"
    dx, mesh_type = 2000, ""
    n_iterations, percent_cutoff = 1200, 87.5
    iter_dt_s = 0.002
    plot, use_fibers, use_best_guess, return_activation_times = 1, 0, 0, 1
    min_n_root_nodes, max_n_root_nodes, root_nodes_dist_apart_um = 6, 10, 5000
    v_endo_min, v_endo_max, v_endo_diff = 80, 140, 10
    v_myo_min, v_myo_max, v_myo_diff = 20, 90, 10
    output_activation_times_dir = None
    ############################################# Best params ##########################################################
    params_best_guess = (85, 40), (911, 1092, 1652, 1660, 11504, 13627, 16022, 17508, 18137)
    params_best_guess = tuple(params_best_guess)
    ####################################################################################################################

    v_endos = list(range(v_endo_min, v_endo_max + 1, v_endo_diff))
    v_myos = list(range(v_myo_min, v_myo_max + 1, v_myo_diff))

    log_every_x_iterations = 1
    log_inf_params = {"main_dir": main_dir, "run_id": run_id, "patient_id": patient_id, "dx": dx, "mesh_type": mesh_type,
                      "n_tries": n_tries, "n_iterations": n_iterations, "percent_cutoff": percent_cutoff,
                      "iter_dt_s": iter_dt_s, "use_fibers": use_fibers, "min_n_root_nodes": min_n_root_nodes,
                      "max_n_root_nodes": max_n_root_nodes, "root_nodes_dist_apart_um": root_nodes_dist_apart_um,
                      "v_endos": v_endos, "v_myos": v_myos, "n_processors": n_processors,
                      "log_every_x_iterations": log_every_x_iterations}

    # Create run directory if needed
    run_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/{run_id}"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    lead_names_to_compare = LEAD_NAMES_12

    print(f"{run_id=}")
    print(f"{lead_names_to_compare=}")
    np.save(f"{run_dir}/running.npy", np.array([1]))
    mother_dir = f"{main_dir}/{inferences_folder}/{benchmark_id}/mother_data"

    # Load target QRS and prepare simulation time to match target time
    leads_target = np.load(f"{main_dir}/{inferences_folder}/{benchmark_id}/mother_data/leads_selected_qrs.npy", allow_pickle=True).item()
    lead_names = list(leads_target.keys())
    times_target_s = leads_target[lead_names[0]][0]
    leads_target_temp = {name: leads_target[name][1] for name in lead_names}
    leads_target = leads_target_temp
    qrs_off_s = max(times_target_s)
    total_time_s = qrs_off_s
    times_s = np.round(np.arange(0, total_time_s + iter_dt_s, iter_dt_s), decimals=6)
    times_s = times_s[times_s <= total_time_s]  # Prevent overstepping beyond total_time_s
    lead_names = list(leads_target.keys())  # Define leads by which exist in target ECG

    # Loading alg mesh and cached geometrical information
    mesh_alg_name = f"{patient_id}_{dx}{mesh_type}.alg"
    mesh_alg_path = f"{main_dir}/Meshes_{dx}/{mesh_alg_name}"
    alg = alg_utils.read_alg_mesh(mesh_alg_path)
    xs, ys, zs, lxs, lys, lzs = alg_utils.unpack_alg_geometry(alg)
    dx = alg_utils.get_dx(xs)
    n_cells = len(xs)
    cache_path = f"{main_dir}/Cache/{patient_id}_{dx}_cache.npy"
    mesh_info_dict = np.load(cache_path, allow_pickle=True).item()

    # Read from cache: endo surface, plane and electrode positions
    keys_to_read = ["endo_labels", "labels_meaning", "electrodes_xyz"]
    endo_labels, labels_meaning, electrodes_xyz = cache.check_cache(mesh_info_dict, keys_to_read)
    lv_val, rv_val = labels_meaning["lv"], labels_meaning["rv"]
    lv_endo_idxs, rv_endo_idxs = np.where(endo_labels[:n_cells] == lv_val)[0], np.where(endo_labels[:n_cells] == rv_val)[0]
    endo_mask = np.zeros(n_cells)
    endo_mask[lv_endo_idxs], endo_mask[rv_endo_idxs] = 1, 1
    endo_idxs = np.where(endo_mask == 1)[0]
    xs_endo, ys_endo, zs_endo = xs[endo_idxs], ys[endo_idxs], zs[endo_idxs]
    alg_endo = alg_utils.alg_from_xs(xs_endo, ys_endo, zs_endo)

    # Preprocessing for pseudo ECG computation
    grid_dict = alg_utils.make_grid_dictionary(xs, ys, zs)
    neighbour_arrays, neighbour_arrays2 = ecg.get_neighbour_arrays(xs, ys, zs, dx, grid_dict)
    elec_grads = ecg.precompute_elec_grads(xs, ys, zs, electrodes_xyz, dx, neighbour_arrays).astype(np.float32)
    grid_endo_dict = alg_utils.make_grid_dictionary(xs_endo, ys_endo, zs_endo)

    root_nodes_neighbour_dist_um = 2 * root_nodes_dist_apart_um
    # Set up candidate root node parameter space
    candidate_root_points, candidate_root_neighbours = qrsm.mesh_subset_with_dist_constraint(alg_endo,
                                                                                           root_nodes_dist_apart_um,
                                                                                           root_nodes_neighbour_dist_um)
    candidate_root_node_indices = []
    xs_candidate, ys_candidate, zs_candidate = [], [], []
    for point in candidate_root_points:
        xs_candidate.append(point[0])
        ys_candidate.append(point[1])
        zs_candidate.append(point[2])

    # candidate_root_node_points contains indices (corresponding to the original alg mesh) of possible root nodes
    flag = [0 for _ in range(len(xs_endo))]
    for point in candidate_root_points:
        flag[grid_endo_dict[point]] = 1
        candidate_root_node_indices.append(grid_dict[point])

    if use_best_guess:
        n_tries, n_iterations = 1, 1  # Only 1 trial needed for best guess
        # Check root indices you are using are within the candidate root node indices list
        for root_index in params_best_guess[1]:
            if root_index not in candidate_root_node_indices:
                raise Exception("Root index in best guess is not in candidate root node indices list, check\
                                root_nodes_dist_apart_um")
        # Only perform dijkstra to the root nodes in use when using best guess
        candidate_root_node_indices = params_best_guess[1]
        # Confine velocity parameter space to best guess velocity parameters
        v_endos, v_myos = [params_best_guess[0][0]], [params_best_guess[0][1]]

    # Used for activation time calculations in dijkstra (need 26 to ensure isotropic propagation is possible)
    adjacency_list_26 = ecg.compute_adjacency_displacement(xs, ys, zs, dx, grid_dict, NEIGHBOURS_26)  # Post-fibers version (displacement vectors for fib projections)

    v_endos.sort()  # Sort v_params (ascending) as inc/decreasing v mutations relies on this
    v_myos.sort()
    v_space = len(v_endos) * len(v_myos)
    print(v_space, "velocity space", flush=True)

    # Parallel computation of all_all_time_matrices
    param_args = [(v_endo, v_myo, adjacency_list_26, endo_mask, use_fibers, candidate_root_node_indices)
                  for v_endo in v_endos for v_myo in v_myos]
    batch_size = int(math.ceil(v_space / n_processors))
    batched_param_args = list(qrsm.batcher(param_args, batch_size))

    all_all_time_matrices = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
        results = executor.map(qrsm.compute_time_matrix_batch, batched_param_args)
        for batch_result in results:
            all_all_time_matrices.update(batch_result)

    # Extract subset of target lead at points closely matching simulated lead time
    target_qrs_idxs = ecg.match_sim_and_target_times(times_s, times_target_s)
    target_leads_qrs = {name: leads_target[name][target_qrs_idxs] for name in lead_names}
    # Compute QRS amplitudes for each target lead then normalise them
    target_qrs_amps = {name: np.max(target_leads_qrs[name]) - np.min(target_leads_qrs[name]) for name in lead_names}
    target_leads_qrs_normed = {name: target_leads_qrs[name] / target_qrs_amps[name] for name in lead_names}

    print(f"Max QRS time simulated: {max(times_s) * 1000}ms", flush=True)
    print(len(candidate_root_points), "root node parameter space", flush=True)

    log.log_init_qrs(run_dir, log_inf_params, candidate_root_points, candidate_root_node_indices, times_target_s,
                     leads_target, alg, times_s)

    if not os.path.exists(f"{run_dir}/pop_ids_and_diffs"):
        os.makedirs(f"{run_dir}/pop_ids_and_diffs")

    # Initialise root nodes and conduction velocity parameters
    current_iter_params = qrsm.init_roots_and_vels(n_tries, min_n_root_nodes, max_n_root_nodes,
                                                 candidate_root_node_indices, v_endos, v_myos, params_best_guess,
                                                 use_best_guess)

    mutated_params, all_ids_and_diff_scores = {}, {}
    activation_cutoff_s = total_time_s
    runtimes = []

    # Main iterative refinement of root nodes loop
    for iter_no in range(n_iterations):
        n_tries = len(current_iter_params)
        n_per_batch = int(round(n_tries / n_processors))
        n_per_batch = 1 if n_per_batch == 0 else n_per_batch
        tries = np.arange(n_tries)
        print(f"===================================== {iter_no} =====================================", flush=True)
        print(f"{iter_no}: n_tries = {n_tries}, n_per_batch = {n_per_batch}", flush=True)

        t0 = time.time()

        # Compute electrode signals for simulations required this iteration
        all_electrodes, activation_times_s, *_ = qrsm.batch_ecg_runner_qrs(n_tries, n_per_batch, qrsm.pseudo_ecg_qrs,
                                                                           times_s, qrsm.action_potential_heaviside,
                                                                           electrodes_xyz, elec_grads, dx,
                                                                           activation_cutoff_s, neighbour_arrays,
                                                                           qrs_params=current_iter_params,
                                                                           all_all_time_matrices=all_all_time_matrices)

        population_ids_check, population_ids, ids_and_ecgs_ats_params = {}, {}, {}

        for i_try in tries:
            # Conversion of params simulated this iteration to ids and record that the param_id is in pop already
            param_id = qrsm.hash_qrs_param(current_iter_params[i_try])
            population_ids[i_try] = param_id
            population_ids_check[param_id] = 1

        # Compute pseudo ECG leads from electrodes and compare to monoalg ECG leads
        all_normed_leads_pseudo, population_diff_scores, all_leads_sim = qrsm.analyse_pseudo_electrodes_qrs(all_electrodes,
                                                                                       target_leads_qrs_normed, lead_names_to_compare=lead_names_to_compare)
        t1 = time.time()

        if n_tries != 0:
            print(round((t1 - t0) / n_tries, 4), "seconds elapsed per try", flush=True)

        for i_try, params in current_iter_params.items():  # Store all params, scores & activation times
            v_params, root_indices = params
            root_indices = list(root_indices)
            root_indices.sort()

            store_activation_times = np.round(np.array(activation_times_s[i_try]) * 1000)  # s to ms
            store_activation_times_ms = np.array(store_activation_times, dtype=np.uint16)

            param_id = qrsm.hash_qrs_param((v_params, tuple(root_indices)))
            all_ids_and_diff_scores[param_id] = [population_diff_scores[i_try], iter_no]
            ids_and_ecgs_ats_params[param_id] = [all_leads_sim[i_try], store_activation_times_ms, params]

        # Retrieval of diff scores in the population but not simulated this iteration
        population_params = current_iter_params.copy()
        new_key = max(population_diff_scores.keys()) + 1

        for key, params in mutated_params.items():  # mutated_params is of the population size
            params = params[0], tuple(params[1])
            param_id = qrsm.hash_qrs_param(params)

            # current_iter_params is of size n_tries of this iteration (unseen params)
            if param_id in all_ids_and_diff_scores and param_id not in population_ids_check:
                # Retrieving seen difference scores kept in the population but that were not simulated this iteration
                population_diff_scores[new_key] = all_ids_and_diff_scores[param_id][0]
                population_params[new_key] = params
                population_ids[new_key] = param_id
                population_ids_check[param_id] = 1
                new_key += 1

        # Find the keys with scores less than or equal to the n th percentile value
        scores = list(population_diff_scores.values())

        percentile_thresh = np.percentile(scores, percent_cutoff)
        keys_below = [key for key, value in population_diff_scores.items() if value <= percentile_thresh]  # Better
        keys_above = [key for key, value in population_diff_scores.items() if value > percentile_thresh]  # Worse

        # Perform mutations on root indices # worse, better keys
        mutated_params = qrsm.mutate_population_activation_params(keys_above, keys_below, population_params, alg,
                                                                  grid_dict, candidate_root_node_indices,
                                                                  candidate_root_neighbours, v_endos, v_myos,
                                                                  all_ids_and_diff_scores)
        next_iter_params = {}

        # Check which new root indices + velocity params have been analysed before already + set up next iteration
        next_iter_tries_ct = 0
        for key, params in mutated_params.items():
            v_params, root_indices = params
            param_id = qrsm.hash_qrs_param((v_params, tuple(root_indices)))

            if param_id not in all_ids_and_diff_scores:  # Only simulate unseen params
                next_iter_params[next_iter_tries_ct] = params
                next_iter_tries_ct += 1

        # Update which root indices and v params to analyse next
        current_iter_params = next_iter_params

        # Find best scores and params this iteration
        min_diff_score = min(population_diff_scores.values())
        min_i_try = min(population_diff_scores, key=population_diff_scores.get)
        min_key = population_params[min_i_try]
        best_params = population_params[min_i_try]
        hash_best_param = qrsm.hash_qrs_param(best_params)

        if hash_best_param in ids_and_ecgs_ats_params:
            # Then the new best reg params were found this iteration (so update all best values)
            best_ats = ids_and_ecgs_ats_params[hash_best_param][1]
            best_leads = ids_and_ecgs_ats_params[hash_best_param][0]

        if iter_no % save_best_every_x == 0:  # Saving of fast results
            alg = alg[:6]
            alg.append(best_ats)
            fast_download_folder = f"fast_{benchmark_id}"
            alg_utils.save_alg_mesh(f"{run_dir}/{fast_download_folder}/bestguess_best_params_{iter_no}.alg", alg)

        runtime_end = time.time()
        runtime_current_total = runtime_end - runtime_start
        runtimes.append(runtime_current_total)

        log.log_progress_qrs(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                                       population_ids, population_diff_scores, ids_and_ecgs_ats_params,
                                       population_params)

        print(len(all_ids_and_diff_scores), "Unique parameter sets tested so far", flush=True)
        print("Best Params:", min_key, flush=True)
        print("Min:", min_diff_score, flush=True)
        # End of iteration loop

    if return_activation_times:
        alg.append(activation_times_s[0])  # Zeroth try of zeroth batch
        mesh_type = "activation_times"
        mesh_alg_name = f"{patient_id}_{dx}_{mesh_type}.alg"

        if output_activation_times_dir is None:
            alg_utils.save_alg_mesh(f"{run_dir}/{mesh_alg_name}", alg, True)
        else:
            alg_utils.save_alg_mesh(f"{output_activation_times_dir}/{mesh_alg_name}", alg, True)

        save_ecg = np.array([times_s, np.array(all_normed_leads_pseudo[0])], dtype=object)
        np.save(f"{run_dir}/py_ecg.npy", save_ecg)

    if plot:
        ecg.plot_ecg([times_s, times_target_s[target_qrs_idxs]], [all_normed_leads_pseudo[0], target_leads_qrs_normed],
                    colors=["red", "black"], xlims=[0, 0.45])

if __name__ == '__main__':
    main()


