"""Runs the model training pipeline.

The script config defines the input features used by the model, its architecture,
and the training hyperparameters. The pipeline has its associated MLFlow run.
"""


import omegaconf
import hydra
import torch

from advanced_data_mining.data import ds_loading


@hydra.main(version_base=None, config_path="cfg", config_name="train_model")
def main(cfg: omegaconf.DictConfig):
    """Runs the model training pipeline."""


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
