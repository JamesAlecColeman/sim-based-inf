import alg_utils
import ecg
import log_analysis_functions as laf
import matplotlib.pyplot as plt
import addcopyfighandler
import os
import compare_distributions as comp
from constants import *
import random

main_dir = "C:/Users/jammanadmin/Documents/sim-based-inf-data"

inferences_folder, repol, save_analysis = "Inferences_twave_local", True, True
patient_id_select, patient_id_skip, run_id_select = None, None, None
stop_thresh, force_iter_final = 0.00002, "max"

compare_to_truth, benchmarks_folder = True, "New_Benchmarks_APDs"
iter_step, x_best = 50, 10  # View approximate convergence every iter_step iterations, just the x_best solutions

coarse_dx = 2000

inferences_path = f"{main_dir}/{inferences_folder}"
targets_in_inf_folder, runs_in_targets = laf.find_inference_runs(inferences_path)
lead_names = LEAD_NAMES_12

if save_analysis:
    analysis_dir = f"{main_dir}/{inferences_folder}/analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

# Prepare figure to plot iteration-wise scores and comparisons to ground truths
width_px, height_px, dpi = 850, 750, 100
width_in, height_in = width_px / dpi, height_px / dpi
fig, axs = plt.subplots(3, 2, figsize=(width_in, height_in), dpi=dpi)
n_rows, n_cols = axs.shape

all_run_scores, all_run_corrs, all_run_absdiffs = {}, {}, {}

n_targ = len(targets_in_inf_folder)
for i_targ, target in enumerate(runs_in_targets.keys()):  # E.g. now in "Inferences_Folder/DTI003_500_ctrl"

    if patient_id_select is not None:
        if target.split("_")[0] != patient_id_select:
            continue

    if patient_id_skip is not None:
        if target.split("_")[0] == patient_id_skip:
            continue

    print(f"{target=}")

    target_fields = target.split("_")
    patient_id, fine_dx, mesh_type = target_fields[0], target_fields[1], target_fields[2]
    n_runs = len(runs_in_targets[target])

    mother_data_path = f"{inferences_path}/{target}/mother_data"
    mother_data_dir = list(os.listdir(mother_data_path))

    truth_times_ms, truth_apd90s_ms = None, None

    if compare_to_truth:
        # Load ground truth activation/repolarisation sequence for this target
        benchmark_alg_path = f"{main_dir}/{benchmarks_folder}/{patient_id}_{coarse_dx}_{mesh_type}_APDs.alg"
        benchmark_alg = alg_utils.read_alg_mesh(benchmark_alg_path)  # APD90s, activation times, repolarisation times
        truth_apd90s_ms, truth_activations_s, truth_repols_ms = benchmark_alg[6], benchmark_alg[7], benchmark_alg[8]
        truth_activations_ms =  truth_activations_s * 1000

        activation_times_count = 0
        for filename in mother_data_dir:
            if "activation_times" in filename and filename[-4:] == ".alg":
                activation_times_count += 1

        print(f"{activation_times_count} activation files in mother data, {n_runs=}")

        truth_times_ms = truth_activations_ms if not repol else truth_repols_ms


    for i_run, run_id in enumerate(runs_in_targets[target]):  # E.g. now in ""DTI003_500_ctrl/runtime_512_-10.0""
        if run_id_select is not None:
            if run_id != run_id_select:
                continue

        run_path = f"{inferences_path}/{target}/{run_id}"
        print("=======================================================================================")
        print(f"{target}/{run_id=}")

        i_iter_maximum = laf.get_max_i_iter(run_path, prefix="all_ids_and_diff_scores")
        all_ids_and_diff_scores = np.load(f"{run_path}/all_ids_and_diff_scores_{i_iter_maximum}.npy", allow_pickle=True).item()
        iterations, n_iterations, log_every_x_iterations = laf.get_iteration_nos(run_path, i_population_name="ids_and_rts_and_ecgs")

        # Application of stopping condition
        (i_iter_final, min_diff_score, median_diff_score, best_params_reg, min_reg_score, median_reg_score,
         abs_moving_avg) = laf.apply_stop_condition(run_path, iterations, twave_diff_threshold=stop_thresh,
                                                    i_population_name="pop_ids_and_diffs/population_ids_and_diff_scores",
                                                    repol=repol, force_iter_final=force_iter_final)

        print(f"Stopped at iteration {i_iter_final} of {n_iterations} iterations")
        print(f"min diff, reg scores = {float(min_diff_score)}, {float(min_reg_score)}")


        # Extract best model at final iteration
        best_x_times, best_x_reg_scores, best_x_leads = laf.get_best_x_rts_or_ats(run_path, i_iter_final, 1, all_ids_and_diff_scores,
                                                                              repol=repol)

        activation_ms = None

        if repol:  # Load activation times from mother dir
            #select_activation = run_id.split("_")[-1]  # When using angle
            select_activation = ""  # When not using angle
            alg_activation = alg_utils.read_alg_mesh(f"{mother_data_path}/{patient_id}_{coarse_dx}_activation_times{select_activation}.alg")
            activation_s = alg_activation[-1]
            activation_ms = activation_s * 1000
            print(f"Run {run_id} using {patient_id}_{coarse_dx}_activation_times{select_activation}.alg as activation used")

            """bestqrsparams = np.load(f"{mother_data_path}/{target}_bestqrsparams_None.npy", allow_pickle=True)
            print(f"{bestqrsparams=}")"""

        # Find iteration-wise scores & comparisons to truths to plot convergence
        iters, iter_scores, iter_median_corrs, iter_median_absdiffs = [], [], [], []

        iter_median_corrs_apds, iter_median_absdiffs_apds = [], []

        corrs, mean_absdiffs, corrs_apds, mean_absdiffs_apds = None, None, None, None

        for iter_no in range(0, i_iter_maximum, iter_step):
            best_x_times, best_x_reg_scores, best_x_leads = laf.get_best_x_rts_or_ats(run_path, iter_no, x_best, all_ids_and_diff_scores,
                                                                                  repol=repol)

            if compare_to_truth:
                # Comparisons with ground truth for the best x solutions this iteration
                corrs = [comp.correlation(times, truth_times_ms) for times in best_x_times]
                mean_absdiffs = [comp.abs_diffs(times, truth_times_ms)[1] for times in best_x_times]

            if repol:  # Calculate and compare APDs
                best_x_apds = [repol_times - activation_ms for repol_times in best_x_times]
                if compare_to_truth:
                    corrs_apds = [comp.correlation(apds, truth_apd90s_ms) for apds in best_x_apds]
                    mean_absdiffs_apds = [comp.abs_diffs(apds, truth_apd90s_ms)[1] for apds in best_x_apds]

            # Store iteration-wise scores, correlations and absolute differences
            iters.append(iter_no)
            iter_scores.append(np.median(best_x_reg_scores))

            if compare_to_truth:
                iter_median_corrs.append(np.median(corrs))
                iter_median_absdiffs.append(np.median(mean_absdiffs))

                if repol:
                    iter_median_corrs_apds.append(np.median(corrs_apds))
                    iter_median_absdiffs_apds.append(np.median(mean_absdiffs_apds))

        # Same color for the same run
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        n_colors = len(color_cycle)
        color_index = (i_run * n_targ + i_targ) % n_colors
        col = color_cycle[color_index]

        axs[0, 0].plot(iters, iter_scores, label=f"{target}_{run_id}", color=col)  # Scores
        axs[0, 0].axvline(x=i_iter_final, linestyle="--", color=col, linewidth=1.0)  # Stopping iteration

        if compare_to_truth:
            axs[1, 0].plot(iters, iter_median_corrs, label=run_id, color=col)  # Correlations
            axs[1, 1].plot(iters, iter_median_absdiffs, label=run_id, color=col)  # Absolute differences

            if repol:
                axs[2, 0].plot(iters, iter_median_corrs_apds, label=run_id, color=col)  # Correlations
                axs[2, 1].plot(iters, iter_median_absdiffs_apds, label=run_id, color=col)  # Absolute differences

        # Prepare to plot final ECG match to target
        times_s, times_target_s = np.load(f"{run_path}/times_s.npy"), np.load(f"{run_path}/times_target_s.npy")
        leads_target = np.load(f"{run_path}/leads_target.npy", allow_pickle=True).item()


        leads_sim_best = best_x_leads[0]

        if repol:
            # QRS amplitudes need calculating specifically from the QRS subset of the target leads
            leads_selected_qrs = np.load(f"{run_path}/leads_selected_qrs.npy", allow_pickle=True).item()
            target_qrs_times_s = leads_selected_qrs["I"][0]
            target_qrs_times_ms = target_qrs_times_s * 1000
            target_qrs_leads = {lead_name: leads_selected_qrs[lead_name][1] for lead_name in lead_names}
            target_leads_amps = {name: np.max(target_qrs_leads[name]) - np.min(target_qrs_leads[name]) for name in lead_names}
        else:
            # QRS amplitudes taken from general target leads because target only contains QRS
            target_leads_amps = {name: np.max(leads_target[name]) - np.min(leads_target[name]) for name in lead_names}

        # Normalisation of QRSes for comparison
        target_leads_normed = {name: leads_target[name] / target_leads_amps[name] for name in lead_names}

        sim_leads_amps = {name: np.max(leads_sim_best[name]) - np.min(leads_sim_best[name]) for name in lead_names}
        sim_leads_normed = {name: leads_sim_best[name] / sim_leads_amps[name] for name in lead_names}

        # Note leads_target has all its original time points saved, not necessarily matched to leads_sim_best
        # So re-match time points
        target_qrs_idxs = ecg.match_sim_and_target_times(times_s, times_target_s)
        leads_target_normed = {name: target_leads_normed[name][target_qrs_idxs] for name in lead_names}
        times_target_s = times_s  # Because match_sim_and_target_times is matching targ times to sim times

        # PLOTTING OF LEAD-SPECIFIC NORMALISATION ECG
        ecg.plot_ecg([times_s, times_target_s],
                     [sim_leads_normed, leads_target_normed],
                     xlims=[0, 0.45], colors=["red", "black"], fig_no=i_targ * len(runs_in_targets[target]) + i_run + 1, title=target+run_id, show=False,
                     labels=["Inferred", "Target"])

        ecg_fig_no = random.randint(1_000_000, 9_999_999)

        # Finding iter nos of where times + ECGs are stored for best params
        iter_nos_to_pop_ids = laf.ids_to_storage_iter_nos([best_params_reg], all_ids_and_diff_scores)

        # Load times of the best x ids from where they were saved before
        best_x_ids_to_rts, best_x_ids_to_params, best_x_ids_to_leads = {}, {}, {}
        for iter_no2, ids in iter_nos_to_pop_ids.items():
            ids_and_rts_and_ecgs_temp = np.load(f"{run_path}/ids_and_rts_and_ecgs_{iter_no2}.npy",
                                                allow_pickle=True).item()
            for id in ids:
                best_x_ids_to_params[id] = ids_and_rts_and_ecgs_temp[id][2]
        best_x_params = [best_x_ids_to_params[id] for id in [best_params_reg]]

        print(f"{best_x_params=}")
        final_times_ms = best_x_times[0]

        if compare_to_truth:
            corr_final = comp.correlation(final_times_ms, truth_times_ms)
            print(f"{run_id} {corr_final=}")
            mean_absdiffs_final = comp.abs_diffs(final_times_ms, truth_times_ms)[1]
            all_run_corrs[f"{target}/{run_id}"] = corr_final
            all_run_absdiffs[f"{target}/{run_id}"] = mean_absdiffs_final

            if repol:
                final_apds_ms = final_times_ms - activation_ms
                corr_apd_final = comp.correlation(final_apds_ms, truth_apd90s_ms)
                print(f"{corr_apd_final=}")

        all_run_scores[f"{target}/{run_id}"] = min_reg_score

        if save_analysis:

            alg = alg_utils.read_alg_mesh(f"{main_dir}/Meshes_{coarse_dx}/{patient_id}_{coarse_dx}.alg")
            alg = alg[:6]
            alg.append(final_times_ms)
            if repol:
                final_apd90s_ms = best_x_times[0] - activation_ms
                alg.append(final_apd90s_ms)
            alg_utils.save_alg_mesh(f"{analysis_dir}/{target}_{run_id}_best.alg", alg)

            if not repol:
                np.save(f"{analysis_dir}/{target}_bestqrsparams_{run_id}.npy", np.array(best_x_params[0], dtype=object))

axs[0, 0].set_ylabel("Scores")
axs[0, 0].legend(fontsize='x-small', borderpad=0.1, labelspacing=0.2, handletextpad=0.2, loc='best')

if compare_to_truth:
    activn_repoln_string = "ATs" if not repol else "RTs"

    axs[1, 0].set_ylabel(f"{activn_repoln_string} Spearman r")
    axs[1, 0].set_ylim([0, 1])

    axs[1, 1].set_ylabel(f"{activn_repoln_string} Abs Diffs (ms)")

    if repol:
        axs[2, 0].set_ylabel("APD90s Spearman r")
        axs[2, 0].set_ylim([0, 1])
        axs[2, 1].set_ylabel("APD90s Abs Diffs (ms)")

# Plot style choices to apply to all subplots
for i in range(n_rows):
    for j in range(n_cols):
        axs[i, j].tick_params(axis='both', which='both', length=8, direction='inout', labelsize=12)
        axs[i, j].xaxis.label.set_fontsize(16)
        axs[i, j].yaxis.label.set_fontsize(16)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['bottom'].set_visible(True)
        axs[i, j].spines['left'].set_visible(True)
        axs[i, j].grid(True, linestyle="--", linewidth=0.5, color="gray", alpha=0.5)
        axs[i, j].set_xlabel("Iterations")

plt.tight_layout()
plt.show()

# At-a-glance summary of run final results as a bar plot
width_px, height_px, dpi = 850, 750, 100
width_in = width_px / dpi
height_in = height_px / dpi
fig, axs = plt.subplots(3, 2, figsize=(width_in, height_in), dpi=dpi)
n_rows, n_cols = axs.shape
axs = axs.flatten()

# Data sources and titles
data_sources = [
    ('Scores per Run', all_run_scores),
    ('Correlations per Run', all_run_corrs),
    ('Absolute Differences per Run', all_run_absdiffs)
]

# Plot bar charts in first 3 subplots
for i, (title, data) in enumerate(data_sources):
    labels = list(data.keys())
    values = list(data.values())
    axs[i].bar(labels, values, color='skyblue')
    axs[i].set_title(title)
    axs[i].set_xlabel('Run ID')
    axs[i].set_ylabel('Value')
    axs[i].tick_params(axis='x', rotation=20, labelsize=8)

for j in range(len(data_sources), len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.show()