"""Runs model testing for a selected MLflow run checkpoint."""
import logging
import pathlib

import hydra
import mlflow
import omegaconf

from advanced_data_mining.experiments import testing
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='test_model')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs testing for the selected model checkpoint."""

    logging_utils.setup_logging('test_model')
    _logger().info('Running model testing with config:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    mlflow_client = mlflow.tracking.MlflowClient(cfg.mlflow_server_uri)

    tester_cfg = testing.ModelTesterConfig.model_validate(
        omegaconf.OmegaConf.to_container(cfg.model_tester_cfg)
    )

    tester = testing.ModelTester(cfg=tester_cfg, mlflow_client=mlflow_client)
    tester.test_model(pathlib.Path(cfg.output_path))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
