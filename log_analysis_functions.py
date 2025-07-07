import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_max_i_iter(benchmark_run_dir, prefix = "all_params_and_diff_scores"):

    filenames = [f for f in os.listdir(benchmark_run_dir) if f.startswith(prefix)]

    # In case we find 2 all_params_and_diff_scores (run ended mid-save)
    possible_iters = []
    for f in filenames:
        possible_iters.append(int(f.split("_")[-1][:-4]))

    i_iter_maximum = min(possible_iters)
    return i_iter_maximum


def get_iteration_nos(benchmark_run_dir, i_population_name="population_params_and_diff_scores"):
    # Extract iteration numbers from the population filenames
    iterations = []
    i_population_filenames = utils.find_files(benchmark_run_dir, i_population_name)
    for filename in i_population_filenames:
        iterations.append(int(filename.split("_")[-1][:-4]))
    iterations.sort()

    if len(iterations) > 1:
        log_every_x_iterations = iterations[1] - iterations[0]
    else:
        log_every_x_iterations = 1

    n_iterations = max(iterations)
    return iterations, n_iterations, log_every_x_iterations


def get_scores(benchmark_run_dir, i_iter, i_population_name="population_params_and_diff_scores", repol=True):
    i_population_params_and_diff_scores = np.load(f"{benchmark_run_dir}/{i_population_name}_{i_iter}.npy",
                                                  allow_pickle=True)
    pop_params, pop_diff_scores = i_population_params_and_diff_scores[0], i_population_params_and_diff_scores[1]
    pop_reg_scores = i_population_params_and_diff_scores[2]

    if not repol:
        pop_reg_scores = pop_diff_scores

    i_tries = sorted(pop_params)#list(pop_params.keys())  # Keys in pop_params and pop_diff_scores dicts

    pop_params_list, pop_diff_scores_list, pop_reg_scores_list = [], [], []

    for i_try in i_tries:
        pop_params_list.append(pop_params[i_try])
        pop_diff_scores_list.append(pop_diff_scores[i_try])
        pop_reg_scores_list.append(pop_reg_scores[i_try])

    min_diff_score = np.min(pop_diff_scores_list)
    i_min_diff_score = np.argmin(pop_diff_scores_list)
    median_diff_score = np.median(pop_diff_scores_list)
    best_params_diff = pop_params_list[i_min_diff_score]

    min_reg_score = np.min(pop_reg_scores_list)
    i_min_reg_score = np.argmin(pop_reg_scores_list)
    median_reg_score = np.median(pop_reg_scores_list)
    best_params_reg = pop_params_list[i_min_reg_score]

    return min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg


def apply_stop_condition(benchmark_run_dir, iterations, window_size=50, twave_diff_threshold=-0.0001,
                         force_iter_final=None, plot=False,
                         i_population_name="population_params_and_diff_scores", repol=True):

    if force_iter_final is None:
        min_scores, median_scores = [], []

        for i_iter in iterations:
            min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg = get_scores(benchmark_run_dir, i_iter, i_population_name=i_population_name, repol=repol)
            # Using regularised scores
            min_scores.append(min_reg_score)
            median_scores.append(median_reg_score)

        # Moving average over window size
        moving_avg = np.convolve(np.diff(median_scores), np.ones(window_size) / window_size, mode='same')
        abs_moving_avg = np.abs(moving_avg)
        abs_thresh = np.abs(twave_diff_threshold)
        below_threshold_indices = np.where(abs_moving_avg < abs_thresh)[0]


        if plot:
            plt.plot(abs_moving_avg, color="gray")
            plt.axhline(abs_thresh, color="red")
            plt.show()

        print(f"Minimum absolute moving average {min(np.abs(moving_avg)):.15f}")

        if len(below_threshold_indices) == 0:
            raise Exception("Has not converged based on this threshold")

        i_iter_final = min(below_threshold_indices)
    else:
        i_iter_final = force_iter_final

        if force_iter_final == "max":
            i_iter_final = np.max(iterations)
            abs_moving_avg = None

    min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg = get_scores(benchmark_run_dir, i_iter_final, i_population_name=i_population_name, repol=repol)

    return int(i_iter_final), min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score, abs_moving_avg


def ids_to_storage_iter_nos(pop_ids, all_ids_and_diff_scores):
    # Find which iterations we must load to retrieve ECGs and AT/RTs for the current population ids
    iter_nos_to_pop_ids = defaultdict(list)  # dict {iter_no: [params1, params2, ...]}
    for id in pop_ids:
        iter_where_stored = all_ids_and_diff_scores[id][1]
        iter_nos_to_pop_ids[iter_where_stored].append(id)
    return iter_nos_to_pop_ids


def get_best_x_rts_or_ats(run_dir, iter_no, x_best_indices, all_ids_and_diff_scores, repol=True):
    # pop_ids_and_diff_scores = [population_ids, population_diff_scores, population_reg_scores, population_params]
    pop_ids_and_diff_scores = np.load(f"{run_dir}/pop_ids_and_diffs/population_ids_and_diff_scores_{iter_no}.npy",
                                      allow_pickle=True)

    if repol:  # Use regularised scores
        pop_reg_scores = pop_ids_and_diff_scores[2]
    else:  # Use diffs
        pop_reg_scores = pop_ids_and_diff_scores[1]

    pop_ids = pop_ids_and_diff_scores[0]
    # pop_ids and pop_reg_scores are dicts like {i_try: id} so convert to arrays
    pop_ids, pop_reg_scores = np.array(list(pop_ids.values())), np.array(list(pop_reg_scores.values()))

    # Get indices of the x best scores
    best_x_indices = np.argsort(pop_reg_scores)[:x_best_indices]
    best_x_ids = pop_ids[best_x_indices]
    best_x_reg_scores = pop_reg_scores[best_x_indices]

    # Finding iter nos of where RTs and ECGs saved of best x ids
    iter_nos_to_pop_ids = ids_to_storage_iter_nos(best_x_ids, all_ids_and_diff_scores)

    # Load RTs of the best x ids from where they were saved before
    best_x_ids_to_rts, best_x_ids_to_params, best_x_ids_to_leads = {}, {}, {}

    for iter_no2, ids in iter_nos_to_pop_ids.items():
        ids_and_rts_and_ecgs_temp = np.load(f"{run_dir}/ids_and_rts_and_ecgs_{iter_no2}.npy",
                                            allow_pickle=True).item()

        for id in ids:
            best_x_ids_to_leads[id] = ids_and_rts_and_ecgs_temp[id][0]
            best_x_ids_to_rts[id] = ids_and_rts_and_ecgs_temp[id][1]
            best_x_ids_to_params[id] = ids_and_rts_and_ecgs_temp[id][2]

    best_x_rts = [best_x_ids_to_rts[id] for id in best_x_ids]
    best_x_leads = [best_x_ids_to_leads[id] for id in best_x_ids]

    return best_x_rts, best_x_reg_scores, best_x_leads


def find_inference_runs(inferences_path):
    # Detects the targets e.g. "DTI003_500_ctrl"
    targets_in_inf_folder = list(os.listdir(inferences_path))

    if "analysis" in targets_in_inf_folder:
        targets_in_inf_folder.remove("analysis")

    # Detects inference runs for each target e.g. "DTI003_500_ctrl/runtime_512_-10.0"
    runs_in_targets = defaultdict(list)
    for target in targets_in_inf_folder:
        target_folder_path = f"{inferences_path}/{target}"
        target_folder_dir = list(os.listdir(target_folder_path))

        if "mother_data" in target_folder_dir:
            target_folder_dir.remove("mother_data")

        if len(target_folder_dir):
            runs_in_targets[target] = target_folder_dir

    return targets_in_inf_folder, runs_in_targets