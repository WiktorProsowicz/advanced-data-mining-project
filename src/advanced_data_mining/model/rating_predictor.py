"""Contains definition of Rating Predictor model."""

from typing import Any, Dict

import lightning as pl


class RatingPredictor(pl.LightningModule):
    """Predicts restaurant ratings based on specified input features."""

    def __init__(self,
                 model_cfg: Dict[str, Any],
                 training_cfg: Dict[str, Any],
                 data_cfg: Dict[str, Any]):
        super().__init__()
