"""Runs testing on models corresponding to a given experiment and composes stats summary."""
import itertools
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List

import hydra
import lightning.pytorch as pl
import omegaconf
import torch

from advanced_data_mining.data import ds_loading
from advanced_data_mining.data import experiments_summary
from advanced_data_mining.model import rating_predictor
from advanced_data_mining.utils import logging_utils
from advanced_data_mining.utils import misc as misc_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _save_example_outputs(run: misc_utils.MLRun,
                          data_module: ds_loading.ProcessedDataModule,
                          output_path: str) -> None:

    checkpoint_path = experiments_summary.get_best_checkpoint_path(run)
    model = rating_predictor.RatingPredictor.load_from_checkpoint(checkpoint_path,  # pylint: disable=no-value-for-parameter
                                                                  map_location=torch.device('cpu'))
    model.eval()

    test_loader = data_module.test_dataloader()

    examples: List[Dict[str, Any]] = []

    for i, inputs in enumerate(itertools.islice(test_loader, 30)):

        raw_sample = data_module.test_ds.get_raw_sample(i)

        with torch.no_grad():
            cl_output, _ = model.sanitize_outputs(*model(inputs))

        examples.append({
            'original_info': raw_sample,
            'predicted_rating': int(cl_output[0].item()) + 1
        })

    with open(output_path, 'w', encoding='utf-8') as output_json_f:
        json.dump(examples, output_json_f, indent=4, ensure_ascii=False)


@hydra.main(version_base=None, config_path='cfg', config_name='summarize_experiment')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs testing and summarizes results for a given experiment."""

    logging_utils.setup_logging('summarize_experiment')

    _logger().info('Running experiment summarization with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

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

    pl.seed_everything(cfg.data_cfg.seed)

    data_module = ds_loading.ProcessedDataModule(
        ds_path=cfg.data_cfg.processed_ds_path,
        batch_size=1,
        n_workers=1,
        n_test_samples=cfg.data_cfg.n_test_samples,
        train_val_split=cfg.data_cfg.train_val_split
    )

    data_module.setup('test')

    for metric in cfg.examples_cfg.choose_by_metrics:

        _logger().info('Saving example outputs for metric %s', metric)

        best_run, worst_run = experiments_summary.get_best_and_worst_runs(mlflow_runs, metric)

        examples_dir = os.path.join(cfg.output_path, 'examples', metric)
        os.makedirs(examples_dir, exist_ok=True)

        _save_example_outputs(best_run,
                              data_module,
                              os.path.join(examples_dir, 'best_run.json'))

        _save_example_outputs(worst_run,
                              data_module,
                              os.path.join(examples_dir, 'worst_run.json'))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
