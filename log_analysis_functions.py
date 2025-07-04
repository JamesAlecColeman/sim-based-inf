import os
import utils
import numpy as np
import matplotlib.pyplot as plt

def get_max_i_iter(benchmark_run_dir, prefix = "all_params_and_diff_scores"):

    filenames = [f for f in os.listdir(benchmark_run_dir) if f.startswith(prefix)]
    print(f"{benchmark_run_dir=}")
    print(f"{filenames=}")
    print(f"{prefix=}")

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
    #i_tries.sort()

    pop_params_list, pop_diff_scores_list, pop_reg_scores_list = [], [], []

    for i_try in i_tries:
        pop_params_list.append(pop_params[i_try])
        pop_diff_scores_list.append(pop_diff_scores[i_try])
        pop_reg_scores_list.append(pop_reg_scores[i_try])

    """pop_params_list = [pop_params[i_try] for i_try in i_tries]
    pop_diff_scores_list = [pop_diff_scores[i_try] for i_try in i_tries]
    pop_reg_scores_list = [pop_reg_scores[i_try] for i_try in i_tries]"""


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

        #print(f"{min(np.abs(moving_avg))=}")
        print(f"Minimum absolute moving average {min(np.abs(moving_avg)):.15f}")

        #print(moving_avg)

        if len(below_threshold_indices) == len(moving_avg):

            raise Exception("Has not converged based on this threshold")


        i_iter_final = min(below_threshold_indices)
    else:
        i_iter_final = force_iter_final

        if force_iter_final == "max":
            i_iter_final = np.max(iterations)
            abs_moving_avg = None

    min_diff_score, median_diff_score, best_params_diff, min_reg_score, median_reg_score, best_params_reg = get_scores(benchmark_run_dir, i_iter_final, i_population_name=i_population_name, repol=repol)

    return int(i_iter_final), min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score, abs_moving_avg