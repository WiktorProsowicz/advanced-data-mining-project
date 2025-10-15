"""Script to perform exploratory data analysis (EDA) on Google Maps reviews dataset."""

import os
import json
import logging

import hydra
import omegaconf

from advanced_data_mining.data import eda


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="perform_eda")
def main(script_cfg: omegaconf.DictConfig):
    """Performs exploratory data analysis (EDA) on Google Maps reviews dataset."""

    _logger().info("Starting EDA script with config:\n%s",
                   omegaconf.OmegaConf.to_container(script_cfg))

    os.makedirs(script_cfg.output_dir, exist_ok=True)

    feature_extractor = eda.EDAFeatureExtractor(raw_ds_path=script_cfg.raw_ds_dir)

    _logger().info("Extracting basic statistics from the dataset...")
    stats = feature_extractor.extract_basic_stats()

    with open(os.path.join(script_cfg.output_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
