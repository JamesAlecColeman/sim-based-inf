import numpy as np
import alg_utils
import os

def log_init_qrs(run_dir, log_inf_params, candidate_root_points, candidate_root_node_indices, times_target_s,
                 leads_target, alg, times_s):
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