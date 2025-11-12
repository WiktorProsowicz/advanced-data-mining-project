"""Contains utilities for summarizing MLflow experiments."""

from typing import Any, Dict
import sys
import logging
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt

from advanced_data_mining.utils import misc as misc_utils


def _logger():
    return logging.getLogger(__name__)


def extract_test_metrics(mlflow_run: misc_utils.MLRun) -> Dict[str, float]:
    """Extracts test metrics from an MLflow run."""

    metrics = {}

    metrics_path = misc_utils.os.path.join(mlflow_run.path, 'metrics', 'test')

    for metric_file in misc_utils.os.listdir(metrics_path):
        with open(misc_utils.os.path.join(metrics_path, metric_file), 'r') as f:
            _, value, _ = f.readline().strip().split(' ')

        metrics[metric_file] = float(value)

    return metrics


def plot_metric_pair(mlflow_runs: list[misc_utils.MLRun],
                     x_metric: str,
                     y_metric: str) -> plt.Figure:
    """Creates a scatter plot for two given metrics across multiple MLflow runs."""

    fig, ax = plt.subplots(figsize=(8, 6))

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

    return fig


def extract_basic_info(mlflow_run: misc_utils.MLRun) -> Dict[str, Any]:
    """Extracts basic information from an MLflow run."""

    info: Dict[str, Any] = {}

    with open(os.path.join(mlflow_run.path, 'metrics', 'val', 'cl_accuracy'), 'r') as f:
        accuracies = [float(value) for _, value, _ in
                      (line.strip().split(' ') for line in f.readlines())]

        best_epoch = np.argmax(accuracies)

    info['best_epoch'] = int(best_epoch)
    info['best_val_cl_accuracy'] = float(accuracies[best_epoch])

    info['bow_encoders_used'] = os.listdir(os.path.join(mlflow_run.path,
                                                        'params', 'model_cfg', 'bow_encoders'))

    with open(os.path.join(mlflow_run.path, 'params', 'optimizer_cfg', 'lr'), 'r') as f:
        info['learning_rate'] = float(f.readline().strip())

    return info


def compose_summary_table(mlflow_runs: list[misc_utils.MLRun],
                          metrics: list[str]) -> str:
    """Composes a summary table of specified metrics across multiple MLflow runs."""

    basic_info_labels = ['best_epoch', 'best_val_cl_accuracy', 'bow_encoders_used', 'learning_rate']

    header = '| Run name | ' + ' | '.join(basic_info_labels + metrics) + ' |\n'
    separator = '| --- ' + '| --- ' * (len(basic_info_labels) + len(metrics)) + ' |\n'
    rows = ''

    mlflow_runs = sorted(mlflow_runs, key=lambda run: extract_test_metrics(run)['cl_accuracy'],
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
