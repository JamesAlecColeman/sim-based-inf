import os
import utils
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_max_i_iter(benchmark_run_dir, prefix = "all_params_and_diff_scores"):
    """Get the maximum completed iteration index from inference run

    Args:
        benchmark_run_dir (str): Path to inference run dir
        prefix (str, optional): Filename prefix to match files containing iteration data.
                                Default is "all_params_and_diff_scores"

    Returns:
        i_iter_maximum (int): Maximum iteration index found based on filenames
    """
    filenames = [f for f in os.listdir(benchmark_run_dir) if f.startswith(prefix)]

    # In case we find 2 all_params_and_diff_scores (run ended mid-save)
    possible_iters = []
    for f in filenames:
        possible_iters.append(int(f.split("_")[-1][:-4]))

    i_iter_maximum = min(possible_iters)
    return i_iter_maximum


def get_iteration_nos(benchmark_run_dir, i_population_name="population_params_and_diff_scores"):
    """Extract iteration numbers from inference run

    Args:
        benchmark_run_dir (str): Path to inference run dir
        i_population_name (str, optional): Prefix of the filenames to search for iteration nos
                                           Default is "population_params_and_diff_scores".

    Returns:
        iterations (list of int): Sorted list of iteration indices found in the filenames.
        n_iterations (int): Maximum iteration number found.
        log_every_x_iterations (int): Estimated logging frequency between saved iterations.
    """
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
    """Load diff and reg scores alongside params for a given iteration

    Args:
        benchmark_run_dir (str): Path to inference run dir
        i_iter (int): Iteration number to load.
        i_population_name (str, optional): Filename prefix for the population file.
                                           Default is "population_params_and_diff_scores".
        repol (bool, optional): Whether to use regularised scores (True) or raw diff scores (False).
                                Default is True.

    Returns:
        min_diff_score (float): Minimum unregularised (diff) score in the population.
        median_diff_score (float): Median unregularised (diff) score in the population.
        best_params_diff (array-like): Parameters corresponding to the minimum diff score.
        min_reg_score (float): Minimum regularised score (or diff score if `repol` is False).
        median_reg_score (float): Median regularised score (or diff score if `repol` is False).
        best_params_reg (array-like): Parameters corresponding to the minimum regularised score.
    """
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
    """Determine whether inference has converged based on stopping condition

    Applies a moving average over median regularised scores and checks whether
    the change has fallen below a specified threshold.

    Args:
        benchmark_run_dir (str): Path to inference run
        iterations (list of int): List of iteration numbers to evaluate.
        window_size (int, optional): Window size for moving average calculation. Default is 50.
        twave_diff_threshold (float, optional): Threshold for the moving average of score differences
                                                (absolute value). Default is -0.0001.
        force_iter_final (int or str, optional): Force use of a specific final iteration.
                                                 Use "max" to select the maximum available iteration.
        plot (bool, optional): Whether to display a plot of the moving average. Default is False.
        i_population_name (str, optional): Filename prefix of population score files.
                                           Default is "population_params_and_diff_scores".
        repol (bool, optional): Whether to use regularised scores (True) or raw diff scores (False).
                                Default is True.

    Returns:
        i_iter_final (int): Chosen final iteration based on stop condition or override.
        min_diff_score (float): Minimum unregularised score at final iteration.
        median_diff_score (float): Median unregularised score at final iteration.
        best_params_reg (array-like): Parameters corresponding to the best regularised score.
        min_reg_score (float): Minimum regularised score at final iteration.
        median_reg_score (float): Median regularised score at final iteration.
        abs_moving_avg (ndarray or None): Absolute moving average of score differences, or None if not computed.
    """
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
    """Map population IDs to the iteration numbers where their data is stored.

    This identifies which iterations need to be loaded in order to retrieve ECGs and AT/RTs
    corresponding to the given population IDs.

    Args:
        pop_ids (list): List of population IDs whose data is needed.
        all_ids_and_diff_scores (dict): Dictionary mapping ID to a tuple where the second element
                                        is the iteration number it was stored in.

    Returns:
        iter_nos_to_pop_ids (defaultdict): Dictionary mapping iteration numbers to lists of population IDs
                                           that were stored in that iteration.
    """
    iter_nos_to_pop_ids = defaultdict(list)  # dict {iter_no: [params1, params2, ...]}
    for id in pop_ids:
        iter_where_stored = all_ids_and_diff_scores[id][1]
        iter_nos_to_pop_ids[iter_where_stored].append(id)
    return iter_nos_to_pop_ids


def get_best_x_rts_or_ats(run_dir, iter_no, x_best_indices, all_ids_and_diff_scores, repol=True):
    """Retrieve RTs (or ATs) and ECG leads for the top X best-scoring parameter sets.

    Loads saved population data and identifies the top X individuals (based on regularised
    or unregularised scores). Then retrieves their corresponding RTs/ATs and ECG leads
    from storage.

    Args:
        run_dir (str): Path to the run directory containing population and ID mapping files.
        iter_no (int): Iteration number to load population scores from.
        x_best_indices (int): Number of best-scoring individuals to retrieve.
        all_ids_and_diff_scores (dict): Mapping of individual IDs to their storage iteration number
                                        and additional score/parameter data.
        repol (bool, optional): Whether to use regularised scores (True) or unregularised diff scores (False).
                                Default is True.

    Returns:
        best_x_rts (list): RT or AT values for the top X individuals (in order of score).
        best_x_reg_scores (ndarray): Regularised (or unregularised) scores of the top X individuals.
        best_x_leads (list): ECG lead dictionaries for the top X individuals.
    """
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
    """Find inference targets and their runs in a given directory.

    Scans the specified inferences directory to identify target folders and the
    inference runs within each target, excluding specific folders like 'analysis'
    and 'mother_data'.

    Args:
        inferences_path (str): Path to the directory containing inference targets.

    Returns:
        targets_in_inf_folder (list of str): List of target folder names found in the directory,
                                             excluding 'analysis'.
        runs_in_targets (defaultdict(list)): Dictionary mapping each target to a list of
                                             its inference run folder names, excluding 'mother_data'.
    """
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