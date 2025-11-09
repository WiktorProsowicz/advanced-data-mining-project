# -*- coding: utf-8 -*-
"""Script to perform exploratory data analysis (EDA) on Google Maps reviews dataset."""
import json
import logging
import os

import hydra
import omegaconf

from advanced_data_mining.data import eda


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='perform_eda')
def main(script_cfg: omegaconf.DictConfig):
    """Performs exploratory data analysis (EDA) on Google Maps reviews dataset."""

    _logger().info('Starting EDA script with config:\n%s',
                   omegaconf.OmegaConf.to_container(script_cfg))

    os.makedirs(script_cfg.output_dir, exist_ok=True)

    feature_extractor = eda.EDAFeatureExtractor(processed_ds_path=script_cfg.processed_ds_dir)

    _logger().info('Extracting basic statistics from the dataset...')
    stats = feature_extractor.extract_basic_stats()

    with open(os.path.join(script_cfg.output_dir, 'stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

    _logger().info('Generating figures for EDA...')

    for fig_name, fig in feature_extractor.get_figures().items():
        fig_path = os.path.join(script_cfg.output_dir, f'{fig_name}.png')
        fig.savefig(fig_path)

    _logger().info('Saving example reviews from the dataset...')

    example_reviews = feature_extractor.get_example_reviews()

    examples_path = os.path.join(script_cfg.output_dir, 'example_reviews.json')
    with open(examples_path, 'w', encoding='utf-8') as f:
        json.dump(example_reviews, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
