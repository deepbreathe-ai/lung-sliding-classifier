import pandas as pd
import os
import yaml
import matplotlib.pyplot as plt
from src.data.sql_utils import add_date_to_filename
from src.data.utils import refresh_folder

cfg = yaml.full_load(open(os.path.join(os.getcwd(), "config.yml"), 'r'))['EXTERNAL_VAL']


def finetune_results_to_csv(experiment_path=os.getcwd() + cfg['PATHS']['EXPERIMENTS']):
    '''
    Saves finetune results for each trial (and for each fold in that trial) as a csv file.
    :param experiment_path: Full path to fine-tuning experiment files.
    :return Str: Full path to the .csv file containing finetuning results across all trials.
    '''
    trial_results = []
    for result_csv in os.listdir(experiment_path):
        # The experiment folder also contains a subdirectory for plots. Skip over this to get to the actual result csvs.
        if os.path.isdir(os.path.join(experiment_path, result_csv)):
            continue
        trial_results.append(pd.read_csv(os.path.join(experiment_path, result_csv)))

    trial_results = pd.concat(trial_results, ignore_index=True)
    num_trials = cfg['NUM_TRIALS']
    num_data_slices = len(os.listdir(experiment_path)) // num_trials
    ext_data_props = [(n + 1) / num_trials for n in range(num_data_slices)] * num_trials
    ext_data_props = pd.DataFrame(ext_data_props, columns=['ext_train_prop'])
    trial_index = []
    for t in range(num_trials):
        trial_index.extend([t + 1] * num_data_slices)
    trial_index = pd.DataFrame(trial_index, columns=['trial'])
    trial_results = pd.concat([trial_index, ext_data_props, trial_results], axis=1)
    print(trial_results)
    # Adding '_df.csv' to the end of the resulting file name gets rid of permission error. Might want to investigate
    # this further.
    final_path = os.path.join(os.getcwd() + cfg['PATHS']['CSV_OUT'], add_date_to_filename('experiment_results') + '_df.csv')
    trial_results.to_csv(final_path)
    return final_path


def plot_finetune_results(results_csv):
    '''
    Plots finetuning results for metrics specified in the config file (currently loss/sensitivity/specificity). Plots
    results by individual trial along with trial-wise averages for each metric.
    :param results_csv: Full path to .csv file containing fine-tuning results
    '''
    results_df = pd.read_csv(results_csv)
    num_trials = cfg['NUM_TRIALS']
    num_data_slices = len(results_df) // num_trials
    # Set up the x-axis for plotting.
    x = list(results_df['ext_train_prop'])[:num_data_slices]
    plot_metrics = cfg['PLOT_METRICS']
    path_to_plots = os.getcwd() + cfg['PATHS']['PLOTS']
    if cfg['REFRESH_FOLDERS']:
        refresh_folder(path_to_plots)
    # Extract metric values from result df. Generate plots and save them.
    plt_ind = 0
    for metric in plot_metrics:
        metric_values = results_df[metric].values
        y = []
        for i in range(0, len(metric_values), num_data_slices):
            y.append(metric_values[i:i + num_data_slices])
        plt.figure(plt_ind)
        for trial_y in y:
            plt.plot(x, trial_y)
        # Compute trial-wise average metric value
        trial_avgs = []
        for i in range(num_data_slices):
            total = 0
            for j in range(num_trials):
                total += metric_values[i + j * num_data_slices]
            avg = total / num_trials
            trial_avgs.append(avg)
        plt.plot(x, trial_avgs, linestyle='dashed')
        plt.xticks(x)
        plt.xlabel('External Training Data Proportion')
        # Sensitivity and specificity metric values need to be flipped to follow convention with original lung sliding
        # manuscript.
        if metric == 'sensitivity':
            metric = 'specificity'
        elif metric == 'specificity':
            metric = 'sensitivity'
        # Capitalize the name of the metric
        y_axis_label = metric[0].upper() + metric[1:]
        plt.ylabel(y_axis_label)
        plt.title(y_axis_label + ' vs Proportion of External Data as Train Set')
        legend_strs = []
        for i in range(num_trials):
            legend_strs.append('Trial ' + str(i + 1))
        legend_strs.append('Average ' + y_axis_label + ' (Trial-Wise)')
        plt.legend(legend_strs)
        plt.savefig(os.path.join(path_to_plots, add_date_to_filename(metric) + '.jpg'))
        plt_ind += 1

