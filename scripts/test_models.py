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


@hydra.main(version_base=None, config_path='cfg', config_name='test_models')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs testing for the selected model checkpoint."""

    logging_utils.setup_logging('test_models')
    _logger().info('Running model testing with config:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    mlflow_client = mlflow.tracking.MlflowClient(cfg.mlflow_server_uri)

    for tester_cfg in cfg.model_testers_cfgs:

        tester = testing.ModelTester(
            cfg=testing.ModelTesterConfig(run_id=tester_cfg.run_id,
                                          checkpoint_name=tester_cfg.checkpoint_name,
                                          test_ds_path=pathlib.Path(cfg.test_ds_path),
                                          processing_metadata_path=pathlib.Path(
                                              cfg.processing_metadata_path),
                                          batch_size=cfg.batch_size),
            mlflow_client=mlflow_client
        )

        _logger().info('Testing model for run_id=%s and checkpoint=%s',
                       tester_cfg.run_id, tester_cfg.checkpoint_name)

        tester.test_model(pathlib.Path(tester_cfg.output_path))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
