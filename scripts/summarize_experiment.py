"""Runs testing on models corresponding to a given experiment and composes stats summary."""

import os
import logging

import hydra
import omegaconf

from advanced_data_mining.utils import logging_utils
from advanced_data_mining.data import experiments_summary
from advanced_data_mining.utils import misc as misc_utils


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="summarize_experiment")
def main(cfg: omegaconf.DictConfig):
    """Runs testing and summarizes results for a given experiment."""

    logging_utils.setup_logging('summarize_experiment')

    _logger().info('Running experiment summarization with configuration:\n%s',
                   omegaconf.OmegaConf.to_container(cfg))

    os.makedirs(cfg.output_path, exist_ok=True)

    mlflow_runs = misc_utils.get_mlruns(cfg.experiment_name)

    for x_metric, y_metric in cfg.metrics_for_plot_axes:
        fig = experiments_summary.plot_metric_pair(
            mlflow_runs=mlflow_runs,
            x_metric=x_metric,
            y_metric=y_metric
        )

        plot_path = os.path.join(
            cfg.output_path,
            f'scatter_{x_metric}_vs_{y_metric}.svg'
        )

        fig.savefig(plot_path)

        _logger().info('Saved scatter plot of %s vs %s to %s',
                       y_metric, x_metric, plot_path)

    for table in cfg.summary_tables:
        summary_table = experiments_summary.compose_summary_table(
            mlflow_runs=mlflow_runs,
            metrics=table.metrics,
            sort_by=table.sort_by
        )

        summary_path = os.path.join(
            cfg.output_path,
            f'summary_table_{table.name}.md'
        )
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_table)

        _logger().info('Saved summary table to %s', summary_path)

    _logger().info('Generating summary figures...')

    for fig_name, fig in experiments_summary.get_summary_figures(mlflow_runs).items():
        fig_path = os.path.join(cfg.output_path, 'summary_figures', f'{fig_name}.svg')
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        fig.savefig(fig_path)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
