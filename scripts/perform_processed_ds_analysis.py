"""Performs analysis and visualizations on processed datasets."""
import logging
import pathlib

import hydra
import omegaconf

from advanced_data_mining.data.eda import processed_ds_analysis
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='perform_processed_ds_analysis')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Performs exploratory data analysis (EDA) on Google Maps reviews dataset."""

    logging_utils.setup_logging('perform_processed_ds_analysis')

    _logger().info('Starting EDA script with config:\n%s',
                   omegaconf.OmegaConf.to_container(script_cfg))

    analyzer = processed_ds_analysis.ProcessedDatasetAnalyzer(
        processed_ds_path=pathlib.Path(script_cfg.processed_ds_path),
        processing_metadata_path=pathlib.Path(script_cfg.processing_metadata_path)
    )

    base_output_dir = pathlib.Path(script_cfg.output_dir)

    analyzer.save_numerical_feature_distributions(base_output_dir / 'num_features_distributions/')

    analyzer.save_trace_features_stats(base_output_dir / 'trace_features_stats/')

    analyzer.save_word_count_stats(base_output_dir / 'word_count_stats/')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
