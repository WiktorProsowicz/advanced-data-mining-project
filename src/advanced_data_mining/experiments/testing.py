"""Contains utilities for testing trained models on processed datasets."""
import logging
import pathlib

import tqdm
import mlflow
from mlflow.entities import Run as MLflowRun
import pandas as pd
import pydantic
from pydantic import Field
import torch

from advanced_data_mining.data import ds_loading
from advanced_data_mining.model import rating_predictor


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ModelTesterConfig(pydantic.BaseModel):
    """Configuration for model testing workflow."""

    run_id: str = Field(description='MLflow run identifier.')
    checkpoint_name: str = Field(
        description='Selected checkpoint file name from run artifacts.')
    test_ds_path: pathlib.Path = Field(
        description='Path to the processed test dataset directory.')
    processing_metadata_path: pathlib.Path = Field(
        description='Path to dataset processing metadata directory.')


class ModelTester:
    """Loads a trained run checkpoint and evaluates predictions on test data."""

    def __init__(self,
                 cfg: ModelTesterConfig,
                 mlflow_client: mlflow.tracking.MlflowClient) -> None:
        self._cfg = cfg
        self._mlflow_client = mlflow_client
        self._batch_size = 64

    def test_model(self, output_dir: pathlib.Path) -> None:
        """Runs model testing and constructs per-sample predictions dataframe.

        Args:
            output_dir: Directory where test results should be stored.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_ds = self._construct_results_df()

    def _construct_results_df(self) -> pd.DataFrame:
        """Constructs a dataframe with predictions and sample metadata."""

        run = self._mlflow_client.get_run(self._cfg.run_id)

        model = rating_predictor.RatingPredictor.load_from_checkpoint(
            checkpoint_path=(pathlib.Path(run.info.artifact_uri)
                             .joinpath('checkpoints')
                             .joinpath(self._cfg.checkpoint_name)),
            weights_only=False
        )
        model.eval()

        data_module = self._load_data_module(run)

        dataloader = data_module.test_dataloader()
        dataset: ds_loading.ProcessedDataset = dataloader.dataset  # type: ignore

        rows: list[dict[str, object]] = []

        for batch_idx, batch in tqdm.tqdm(enumerate(dataloader),
                                          total=len(dataloader),
                                          desc='Running inference'):

            batch = data_module.transfer_batch_to_device(batch,
                                                         device=model.device,
                                                         dataloader_idx=0)
            with torch.no_grad():
                predictions = model.predict(batch).cpu()

            for sample_offset, predicted_class in enumerate(predictions):
                metadata = dataset.get_raw_sample(batch_idx * self._batch_size + sample_offset)

                row: dict[str, object] = {
                    'rating': int(metadata.review.rating),
                    'predicted_rating': int(predicted_class),
                    'is_translated': bool(metadata.is_translated),
                    'n_words': metadata.n_words,
                    'n_sentences': metadata.n_sentences,
                    'n_author_reviews_index': metadata.n_author_reviews_index,
                }

                if metadata.review.categorized_opinions is not None:
                    for feature_name, feature_value in metadata.review.categorized_opinions.items():
                        row[feature_name] = feature_value

                rows.append(row)

        return pd.DataFrame(rows)

    def _load_data_module(
            self: 'ModelTester',
            run: MLflowRun) -> ds_loading.ProcessedDataModule:
        """Loads data module used to build the test dataloader."""

        ds_cfg_dict: dict[str, str] = {
            key.split('ds_cfg/', 1)[1]: value
            for key, value in run.data.params.items()
            if key.startswith('ds_cfg/')
        }

        data_module = ds_loading.ProcessedDataModule(
            ds_cfg=ds_loading.ProcessedDatasetConfig.model_validate(ds_cfg_dict),
            ds_path=self._cfg.test_ds_path,
            metadata_path=self._cfg.processing_metadata_path,
            batch_size=self._batch_size,
            n_workers=0,
            train_val_split=0.0,
        )

        data_module.setup(stage='test')

        return data_module
