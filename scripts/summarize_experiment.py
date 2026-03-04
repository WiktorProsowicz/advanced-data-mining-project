"""Runs testing on models corresponding to a given experiment and composes stats summary."""
import logging
import pathlib
from typing import List

import hydra
import mlflow
import omegaconf

from advanced_data_mining.data import experiments_summary
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='summarize_experiment')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs testing and summarizes results for a given experiment."""

    logging_utils.setup_logging('summarize_experiment')

    _logger().info('Running experiment summarization with configuration:\n%s',
                   omegaconf.OmegaConf.to_yaml(cfg))

    output_dir = pathlib.Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_client = mlflow.tracking.MlflowClient(cfg.mlflow_server_uri)

    for summarizer_cfg in cfg.experiment_summarizers:
        _logger().info('Processing summarizer for experiment: %s',
                       summarizer_cfg.experiment_name)

        summarizer_config = experiments_summary.ExperimentSummarizerConfig.model_validate(
            omegaconf.OmegaConf.to_container(summarizer_cfg)
        )

        summarizer = experiments_summary.ExperimentSummarizer(summarizer_config, mlflow_client)
        summarizer.summarize(output_dir / summarizer_cfg.experiment_name)

        _logger().info('Completed summarization for experiment: %s',
                       summarizer_cfg.experiment_name)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
