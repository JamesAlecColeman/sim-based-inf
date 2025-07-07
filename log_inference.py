import numpy as np
import alg_utils
import os


def log_init_qrs(run_dir, log_inf_params, candidate_root_points, candidate_root_node_indices, times_target_s,
                 leads_target, alg, times_s):
    """Save initial QRS inference parameters and data

    Args:
        run_dir (str): Directory to save files.
        log_inf_params (dict): Inference parameters to log.
        candidate_root_points (array-like): Candidate root node points
        candidate_root_node_indices (array-like): Indices of candidate root nodes in .alg
        times_target_s (array-like): Target times in seconds.
        leads_target (dict): Lead signals corresponding to target times.
        alg (alg): save mesh
        times_s (array-like): Time points in seconds.
    """
    np.save(f"{run_dir}/log_inf_params.npy", log_inf_params)
    np.save(f"{run_dir}/candidate_root_points.npy", candidate_root_points)
    np.save(f"{run_dir}/candidate_root_node_indices.npy", candidate_root_node_indices)
    np.save(f"{run_dir}/times_target_s.npy", times_target_s)
    np.save(f"{run_dir}/leads_target.npy", leads_target)
    np.save(f"{run_dir}/times_s.npy", times_s)
    np.save(f"{run_dir}/log_inf_params.npy", log_inf_params)
    np.save(f"{run_dir}/log_inf_params.npy", log_inf_params)
    alg_utils.save_alg_mesh(f"{run_dir}/alg.alg", alg)


def log_progress_qrs(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                 population_ids, population_diff_scores, ids_and_ecgs_rts_params,
                 population_params, misc_save=None):
    """Log QRS inference progress by saving runtime and population data at specified iterations.

    Args:
        run_dir (str): Directory where log files are saved.
        iter_no (int): Current iteration number.
        log_every_x_iterations (int): Interval of iterations at which to log data.
        runtimes (list or array): Runtimes recorded for iterations.
        all_ids_and_diff_scores (dict): Dictionary mapping IDs to difference scores.
        population_ids (dict): Population IDs keyed by try number.
        population_diff_scores (dict): Population difference scores keyed by try number.
        ids_and_ecgs_rts_params (dict): Dictionary mapping IDs to ECGs, RTs/ATs, and parameters.
        population_params (dict): Population parameters keyed by trial number.
        misc_save (list, optional): Additional data items to save, if any.
    """
    if iter_no % log_every_x_iterations == 0:
        # Save log files
        np.save(f"{run_dir}/runtimes_{iter_no}.npy", runtimes)
        np.save(f"{run_dir}/all_ids_and_diff_scores_{iter_no}.npy", all_ids_and_diff_scores)
        population_ids_and_diff_scores = [population_ids, population_diff_scores, population_params]

        np.save(f"{run_dir}/pop_ids_and_diffs/population_ids_and_diff_scores_{iter_no}.npy", population_ids_and_diff_scores)

        np.save(f"{run_dir}/ids_and_rts_and_ecgs_{iter_no}.npy", ids_and_ecgs_rts_params)

        if misc_save is not None:

            for i, item in enumerate(misc_save):
                np.save(f"{run_dir}/misc_save_item{i}.npy", item)

        # Remove previous log for runtimes and all_params_and_diff_scores only after saving latest
        previous_save_iter_no = int(iter_no - log_every_x_iterations)
        if previous_save_iter_no >= 0:
            os.remove(f"{run_dir}/runtimes_{previous_save_iter_no}.npy")
            os.remove(f"{run_dir}/all_ids_and_diff_scores_{previous_save_iter_no}.npy")


def log_init_twave(run_dir, log_inf_params, times_target_s, leads_target, alg, times_s, activation_times):
    """Log initial data for T-wave inference before iterations begin.

    Args:
        run_dir (str): Directory to save the log files.
        log_inf_params (dict): Inference parameters to log.
        times_target_s (array): Target times in seconds.
        leads_target (array): Target ECG leads.
        alg (alg): mesh
        times_s (array): Times in seconds corresponding to the alg mesh.
        activation_times (array): Activation times data to save.
    """
    np.save(f"{run_dir}/log_inf_params.npy", log_inf_params)
    np.save(f"{run_dir}/times_target_s.npy", times_target_s)
    np.save(f"{run_dir}/leads_target.npy", leads_target)
    np.save(f"{run_dir}/times_s.npy", times_s)
    np.save(f"{run_dir}/activation_times.npy", activation_times)
    alg_utils.save_alg_mesh(f"{run_dir}/alg.alg", alg)


def log_progress_twave_no_segments(run_dir, iter_no, log_every_x_iterations, runtimes, all_ids_and_diff_scores,
                 population_ids, population_diff_scores, ids_and_ecgs_rts_params, population_reg_scores, misc_save=None):
    """Log progress of T-wave inference without segments at specified iteration intervals.

    Args:
        run_dir (str): Directory to save the log files.
        iter_no (int): Current iteration number.
        log_every_x_iterations (int): Interval of iterations to log progress.
        runtimes (list or array): Recorded runtimes.
        all_ids_and_diff_scores (dict): Dictionary of all IDs and their difference scores.
        population_ids (list): Population parameter IDs.
        population_diff_scores (list): Population difference scores.
        ids_and_ecgs_rts_params (dict): Dictionary mapping IDs to ECGs, RTs, and parameters.
        population_reg_scores (list): Population regularized scores.
        misc_save (list, optional): List of miscellaneous items to save.
    """
    if iter_no % log_every_x_iterations == 0:
        # Save log files
        np.save(f"{run_dir}/runtimes_{iter_no}.npy", runtimes)
        np.save(f"{run_dir}/all_ids_and_diff_scores_{iter_no}.npy", all_ids_and_diff_scores)
        population_ids_and_diff_scores = [population_ids, population_diff_scores, population_reg_scores]
        np.save(f"{run_dir}/pop_ids_and_diffs/population_ids_and_diff_scores_{iter_no}.npy", population_ids_and_diff_scores)


        for id, vals in ids_and_ecgs_rts_params.items():

            apds = vals[2]["apd90s_ms"]
            ids_and_ecgs_rts_params[id][2]["apd90s_ms"] = np.array(apds, dtype=np.int16)

        np.save(f"{run_dir}/ids_and_rts_and_ecgs_{iter_no}.npy", ids_and_ecgs_rts_params)

        if misc_save is not None:

            for i, item in enumerate(misc_save):
                np.save(f"{run_dir}/misc_save_item{i}.npy", item)

        # Remove previous log for runtimes and all_params_and_diff_scores only after saving latest
        previous_save_iter_no = int(iter_no - log_every_x_iterations)
        if previous_save_iter_no >= 0:
            os.remove(f"{run_dir}/runtimes_{previous_save_iter_no}.npy")
            os.remove(f"{run_dir}/all_ids_and_diff_scores_{previous_save_iter_no}.npy")