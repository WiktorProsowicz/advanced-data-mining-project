"""Contains utilities for summarizing MLflow experiments."""
import ast
import collections
import logging
import os
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import matplotlib.figure as mlp_figure
import matplotlib.pyplot as plt
import numpy as np

from advanced_data_mining.utils import misc as misc_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def extract_test_metrics(mlflow_run: misc_utils.MLRun) -> Dict[str, float]:
    """Extracts test metrics from an MLflow run."""

    metrics = {}

    metrics_path = os.path.join(mlflow_run.path, 'metrics', 'test')

    for metric_file in os.listdir(metrics_path):
        with open(os.path.join(metrics_path, metric_file), encoding='utf-8') as f:
            _, value, _ = f.readline().strip().split(' ')

        metrics[metric_file] = float(value)

    return metrics


def plot_metric_pair(mlflow_runs: list[misc_utils.MLRun],
                     x_metric: str,
                     y_metric: str) -> mlp_figure.Figure:
    """Creates a scatter plot for two given metrics across multiple MLflow runs."""

    fig, ax = plt.subplots(figsize=(12, 12))

    runs_metrics = {
        run: extract_test_metrics(run) for run in mlflow_runs
    }

    if not all(x_metric in metrics and y_metric in metrics
               for metrics in runs_metrics.values()):
        _logger().critical('Not all runs contain the specified metrics: %s, %s', x_metric, y_metric)
        sys.exit(1)

    x_values = [metrics[x_metric] for metrics in runs_metrics.values()]
    y_values = [metrics[y_metric] for metrics in runs_metrics.values()]

    ax.scatter(x_values, y_values)

    best_x = min(runs_metrics.items(), key=lambda item: item[1][x_metric])
    best_y = max(runs_metrics.items(), key=lambda item: item[1][y_metric])
    worst_x = max(runs_metrics.items(), key=lambda item: item[1][x_metric])
    worst_y = min(runs_metrics.items(), key=lambda item: item[1][y_metric])

    ax.scatter(best_x[1][x_metric], best_x[1][y_metric], color='green', label=best_x[0].run_name)
    ax.scatter(best_y[1][x_metric], best_y[1][y_metric], color='blue', label=best_y[0].run_name)
    ax.scatter(worst_x[1][x_metric], worst_x[1][y_metric], color='red', label=worst_x[0].run_name)
    ax.scatter(worst_y[1][x_metric], worst_y[1][y_metric],
               color='orange', label=worst_y[0].run_name)

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f'Scatter Plot of {y_metric} vs {x_metric}')
    ax.legend()

    plt.close(fig)

    return fig


def extract_basic_info(mlflow_run: misc_utils.MLRun) -> Dict[str, Any]:
    """Extracts basic information from an MLflow run."""

    info: Dict[str, Any] = {}

    cpt_metric_path = os.path.join(mlflow_run.path, 'metrics', 'val', 'cl_accuracy_weighted_fine')
    with open(cpt_metric_path, encoding='utf-8') as f:
        accuracies = [float(value) for _, value, _ in
                      (line.strip().split(' ') for line in f.readlines())]

        best_epoch = np.argmax(accuracies)

    info['best_epoch'] = int(best_epoch)
    info['n_epochs'] = len(accuracies)
    info['best_val_cl_accuracy'] = float(accuracies[best_epoch])

    info['bow_encoders_used'] = os.listdir(os.path.join(mlflow_run.path,
                                                        'params', 'model_cfg', 'bow_encoders'))

    opt_cfg_path = os.path.join(mlflow_run.path, 'params', 'optimizer_cfg', 'lr')
    with open(opt_cfg_path, encoding='utf-8') as f:
        info['learning_rate'] = float(f.readline().strip())

    model_cfg_path = os.path.join(mlflow_run.path, 'params', 'model_cfg')
    with open(os.path.join(model_cfg_path, 'post_net', 'hidden_dims'), encoding='utf-8') as f:
        hidden_dims = ast.literal_eval(f.readline().strip())
        info['post_net_hidden_dims'] = hidden_dims

    num_enc_path = os.path.join(model_cfg_path, 'numerical_feature_encoder')

    if os.path.isdir(num_enc_path):
        with open(os.path.join(num_enc_path, 'supported_features'), encoding='utf-8') as f:
            supported_features = ast.literal_eval(f.readline().strip())
            info['numerical_features_used'] = supported_features

    else:
        info['numerical_features_used'] = []

    return info


def get_best_checkpoint_path(run: misc_utils.MLRun) -> str:
    """Returns path to checkpoint corresponding with the best metric."""

    basic_info = extract_basic_info(run)

    for epoch_file in os.listdir(os.path.join(run.path, 'checkpoints')):

        if epoch_file.startswith(f'epoch={basic_info["best_epoch"]}-'):
            return os.path.join(run.path, 'checkpoints', epoch_file)

    _logger().critical('Failed to obtain best checkpoint path!')
    sys.exit(1)


def compose_summary_table(mlflow_runs: list[misc_utils.MLRun],
                          metrics: list[str],
                          sort_by: str) -> str:
    """Composes a summary table of specified metrics across multiple MLflow runs."""

    basic_info_labels = ['best_epoch', 'n_epochs', 'best_val_cl_accuracy',
                         'bow_encoders_used', 'numerical_features_used', 'learning_rate']

    header = '| Run name | ' + ' | '.join(basic_info_labels + metrics) + ' |\n'
    separator = '| --- ' + '| --- ' * (len(basic_info_labels) + len(metrics)) + ' |\n'
    rows = ''

    mlflow_runs = sorted(mlflow_runs,
                         key=lambda run: extract_test_metrics(run)[sort_by],
                         reverse=True)

    for run in mlflow_runs:
        run_metrics = extract_test_metrics(run)
        basic_info = extract_basic_info(run)

        row = f'| {run.run_name} | '
        row += ' | '.join(str(basic_info[label]) for label in basic_info_labels) + ' | '
        row += ' | '.join(f'{run_metrics.get(metric, "N/A"):.4f}' for metric in metrics) + ' |\n'

        rows += row

    table = header + separator + rows

    return table


def get_summary_figures(mlflow_runs: list[misc_utils.MLRun]) -> Dict[str, mlp_figure.Figure]:
    """Generates summary figures for given MLflow runs."""

    figures = {}

    runs_metrics = {
        run: extract_test_metrics(run) for run in mlflow_runs
    }

    figures.update(_get_metric_distributions_figures(runs_metrics))

    return figures


def get_best_and_worst_runs(mlflow_runs: list[misc_utils.MLRun],
                            metric: str) -> Tuple[misc_utils.MLRun, misc_utils.MLRun]:
    """Returns best and worst runs with respect to a given metric."""

    runs_metrics = {run: extract_test_metrics(run) for run in mlflow_runs}

    if not runs_metrics:
        _logger().critical('Cannot get best and worst run from an empty sequence!')
        sys.exit(1)

    return (max(mlflow_runs, key=lambda run: runs_metrics[run][metric]),
            min(mlflow_runs, key=lambda run: runs_metrics[run][metric]))


def _get_metric_distributions_figures(
        runs_metrics: Dict[misc_utils.MLRun, Dict[str, float]]) -> Dict[str, mlp_figure.Figure]:

    figures: Dict[str, mlp_figure.Figure] = {}

    encoders_groups = collections.defaultdict(list)
    lr_groups = collections.defaultdict(list)
    postnet_hidden_dims_groups = collections.defaultdict(list)
    numerical_features_groups = collections.defaultdict(list)
    trace_features_groups = collections.defaultdict(list)

    for run in runs_metrics:

        basic_info = extract_basic_info(run)

        encoders = tuple(sorted(basic_info['bow_encoders_used']))
        encoders_groups[encoders].append(run)

        lr_groups[basic_info['learning_rate']].append(run)

        post_net_hidden_dims = tuple(basic_info['post_net_hidden_dims'])
        postnet_hidden_dims_groups[post_net_hidden_dims].append(run)

        valid_num_features = ['num_words', 'num_sentences', 'is_from_cracow']

        numerical_features = tuple(sorted({
            feat for feat in basic_info['numerical_features_used']
            if feat in valid_num_features
        }))
        numerical_features_groups[numerical_features].append(run)

        trace_features = tuple(sorted({
            feat for feat in basic_info['numerical_features_used']
            if feat not in numerical_features
        }))
        trace_features_groups[trace_features].append(run)

    figures.update(_get_metric_distributions_by_groups(
        runs_metrics=runs_metrics,
        groups={', '.join(k): v for k, v in encoders_groups.items()},
        label='distributions_by_bow_encoders'
    ))

    figures.update(_get_metric_distributions_by_groups(
        runs_metrics=runs_metrics,
        groups={str(k): v for k, v in lr_groups.items()},
        label='distributions_by_learning_rate'
    ))

    figures.update(_get_metric_distributions_by_groups(
        runs_metrics=runs_metrics,
        groups={str(k): v for k, v in postnet_hidden_dims_groups.items()},
        label='distributions_by_postnet_hidden_dims'
    ))

    figures.update(_get_metric_distributions_by_groups(
        runs_metrics=runs_metrics,
        groups={', '.join(k): v for k, v in numerical_features_groups.items()},
        label='distributions_by_numerical_features'
    ))

    figures.update(_get_metric_distributions_by_groups(
        runs_metrics=runs_metrics,
        groups={', '.join(k): v for k, v in trace_features_groups.items()},
        label='distributions_by_trace_features'
    ))

    return figures


def _get_metric_distributions_by_groups(
        runs_metrics: Dict[misc_utils.MLRun, Dict[str, float]],
        groups: Dict[str, List[misc_utils.MLRun]],
        label: str
) -> Dict[str, mlp_figure.Figure]:

    metrics = ['cl_accuracy_weighted_fine', 'cl_accuracy_weighted_coarse',
               'cl_accuracy_macro_fine', 'cl_accuracy_macro_coarse']

    figures: Dict[str, mlp_figure.Figure] = {}

    for metric in metrics:

        fig, ax = plt.subplots()

        ax.set_ylabel(metric)
        ax.set_title(f'Distribution of {metric} by BOW Encoders Used')

        values = {bow_group: [runs_metrics[run][metric] for run in runs]
                  for bow_group, runs in groups.items()}

        plot_parts = ax.violinplot(list(values.values()), showmeans=True,
                                   showmedians=False, quantiles=[[.25, .75]] * len(groups))

        plot_parts['cquantiles'].set_color('red')

        ax.set_xticks(np.arange(1, len(values) + 1))
        ax.set_xticklabels(list(values.keys()), rotation=45, ha='right')
        ax.grid(axis='y')

        fig.tight_layout()
        plt.close(fig)

        figures[f'{label}/{metric}'] = fig

    return figures
