"""Script to perform exploratory data analysis (EDA) on Google Maps reviews dataset."""

import logging
import pathlib

import hydra
import omegaconf

from advanced_data_mining.data.eda import raw_eda
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


@hydra.main(version_base=None, config_path='cfg', config_name='perform_eda')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Performs exploratory data analysis (EDA) on Google Maps reviews dataset."""

    logging_utils.setup_logging('perform_eda')

    _logger().info('Starting EDA script with config:\n%s',
                   omegaconf.OmegaConf.to_container(script_cfg))

    output_dir = pathlib.Path(script_cfg.output_dir)
    eda_engine = raw_eda.RawEDA(raw_ds_path=script_cfg.raw_ds_path)

    eda_engine.save_authors_stats(output_dir=output_dir / 'authors_stats/')

    eda_engine.save_review_stats(output_dir=output_dir / 'review_stats/')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
