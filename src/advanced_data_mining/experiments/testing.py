"""Contains utilities for testing trained models on processed datasets."""
import json
import logging
import pathlib
from typing import Annotated
from typing import Any

import mlflow
import pandas as pd
import pydantic
import torch
import tqdm
import yaml
from mlflow.entities import Run as MLflowRun
from pydantic import Field

from advanced_data_mining.data import ds_loading
from advanced_data_mining.data.structs import processed_ds
from advanced_data_mining.model import rating_predictor


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ModelTesterConfig(pydantic.BaseModel):
    """Configuration for model testing workflow."""

    run_id: Annotated[str, Field(description='MLflow run identifier.')]
    checkpoint_name: Annotated[str, Field(
        description='Selected checkpoint file name from run artifacts.')]
    test_ds_path: Annotated[pathlib.Path, Field(
        description='Path to the processed test dataset directory.')]
    processing_metadata_path: Annotated[pathlib.Path, Field(
        description='Path to dataset processing metadata directory.')]
    batch_size: Annotated[int, Field(description='Batch size for testing.')]


class ModelTester:
    """Loads a trained run checkpoint and evaluates predictions on test data."""

    def __init__(self,
                 cfg: ModelTesterConfig,
                 mlflow_client: mlflow.tracking.MlflowClient) -> None:
        self._cfg = cfg
        self._mlflow_client = mlflow_client

        self._run = self._mlflow_client.get_run(self._cfg.run_id)
        self._model = rating_predictor.RatingPredictor.load_from_checkpoint(
            checkpoint_path=(pathlib.Path(self._run.info.artifact_uri)
                             .joinpath('checkpoints')
                             .joinpath(self._cfg.checkpoint_name)),
            weights_only=False
        )
        self._model.eval()

        self._data_module = self._load_data_module(self._run)
        self._dataloader = self._data_module.test_dataloader()
        self._dataset: ds_loading.ProcessedDataset = self._dataloader.dataset  # type: ignore

    def test_model(self, output_dir: pathlib.Path) -> None:
        """Runs model testing and constructs per-sample predictions dataframe.

        Args:
            output_dir: Directory where test results should be stored.
        """
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results_ds = self._construct_results_df()

        setup_to_df = {
            'fine': results_ds,
            'coarse': self._convert_results_to_coarse(results_ds),
        }

        for setup_name, setup_df in setup_to_df.items():
            self._save_distribution_by_category(
                data=setup_df.rename(columns={'is_translated': 'category'}),
                output_dir=output_dir / 'metrics_by_is_translated' / setup_name,
            )
            self._save_distribution_by_category(
                data=setup_df.rename(columns={'n_author_reviews': 'category'}),
                output_dir=output_dir / 'metrics_by_n_author_reviews' / setup_name,
            )

            categorized_cols = [col for col in setup_df.columns if col.startswith('cat_')]
            for cat_col in categorized_cols:
                self._save_distribution_by_category(
                    data=setup_df.rename(columns={cat_col: 'category'}),
                    output_dir=output_dir / f'metrics_by_{cat_col}' / setup_name,
                )

            self._save_global_metrics(setup_df, output_dir / f'global_metrics_{setup_name}.json')

    def _construct_results_df(self) -> pd.DataFrame:
        """Constructs a dataframe with predictions and sample metadata."""

        n_reviews_label_mapping = processed_ds.ProcessingMetadataPathHandler(
            self._cfg.processing_metadata_path).get_n_author_reviews_label_mapping()

        rows: list[dict[str, object]] = []

        for batch_idx, batch in tqdm.tqdm(enumerate(self._dataloader),
                                          total=len(self._dataloader),
                                          desc='Running inference'):

            batch = self._data_module.transfer_batch_to_device(batch,
                                                               device=self._model.device,
                                                               dataloader_idx=0)
            with torch.no_grad():
                predictions = self._model.predict(batch).cpu()

            for sample_offset, predicted_class in enumerate(predictions):
                metadata = self._dataset.get_raw_sample(batch_idx * self._cfg.batch_size +
                                                        sample_offset)

                row: dict[str, object] = {
                    'rating': int(metadata.review.rating),
                    'predicted_rating': int(predicted_class),
                    'is_translated': bool(metadata.is_translated),
                    'n_author_reviews': n_reviews_label_mapping[metadata.n_author_reviews_index],
                }

                if metadata.review.categorized_opinions is not None:
                    for feature_name, feature_value in metadata.review.categorized_opinions.items():
                        row[f'cat_{feature_name}'] = feature_value

                rows.append(row)

        return pd.DataFrame(rows)

    def _save_distribution_by_category(self,
                                       data: pd.DataFrame,
                                       output_dir: pathlib.Path) -> None:
        """Saves precision and recall distribution for the selected category."""

        output_dir.mkdir(parents=True, exist_ok=True)

        summaries: list[dict[str, object]] = []
        labels = sorted(set(data['rating'].astype(int).tolist()) |
                        set(data['predicted_rating'].astype(int).tolist()))

        for category_value, category_df in data.groupby('category', dropna=False):

            summaries.append({
                'category': category_value,
                'n_samples': int(category_df.shape[0]),
                **self._calculate_group_metrics(category_df, labels)
            })

        summary_df = pd.DataFrame(summaries).sort_values('category')

        summary_path = output_dir / 'metrics_by_category.csv'
        summary_df.to_csv(summary_path, index=False)

    def _calculate_group_metrics(self,
                                 category_df: pd.DataFrame,
                                 labels: list[int]) -> dict[str, float]:
        """Calculates class-wise and average precision/recall for one category group."""

        y_true = category_df['rating'].astype(int)
        y_pred = category_df['predicted_rating'].astype(int)

        precision_values: list[float] = []
        recall_values: list[float] = []
        metrics: dict[str, float] = {}

        for label in labels:
            tp = int(((y_true == label) & (y_pred == label)).sum())
            fp = int(((y_true != label) & (y_pred == label)).sum())
            fn = int(((y_true == label) & (y_pred != label)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            precision_values.append(precision)
            recall_values.append(recall)
            metrics[f'precision_{label}'] = float(precision)
            metrics[f'recall_{label}'] = float(recall)

        metrics['precision_avg'] = float(sum(precision_values) / len(precision_values))
        metrics['recall_avg'] = float(sum(recall_values) / len(recall_values))

        return metrics

    def _save_global_metrics(self,
                             results_df: pd.DataFrame,
                             output_path: pathlib.Path) -> None:
        """Saves global metrics and basic information about the results to a JSON file."""

        labels = sorted(set(results_df['rating'].astype(int).tolist()) |
                        set(results_df['predicted_rating'].astype(int).tolist()))

        global_metrics = self._calculate_group_metrics(results_df, labels)

        global_info = {
            'n_samples': int(results_df.shape[0]),
            'n_classes': len(labels),
            'class_labels': labels,
            **global_metrics
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(global_info, f, indent=2)

    def _convert_results_to_coarse(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Maps fine-grained ratings to coarse classes using model's thresholding logic."""

        coarse_df = results_df.copy()

        fine_true = coarse_df['rating'].astype(int) - 1
        fine_pred = coarse_df['predicted_rating'].astype(int) - 1

        coarse_df['rating'] = (fine_true >= 3).astype(int)
        coarse_df['predicted_rating'] = (fine_pred >= 3).astype(int)

        return coarse_df

    def _load_data_module(
            self: 'ModelTester',
            run: MLflowRun) -> ds_loading.ProcessedDataModule:
        """Loads data module used to build the test dataloader."""

        ds_cfg_dict: dict[str, Any] = {
            key.split('ds_cfg/', 1)[1]: value
            for key, value in run.data.params.items()
            if key.startswith('ds_cfg/')
        }

        ds_cfg_dict = yaml.safe_load('\n'.join(f'{key}: {value}'
                                               for key, value in ds_cfg_dict.items())
                                     .replace('None', 'null'))

        data_module = ds_loading.ProcessedDataModule(
            ds_cfg=ds_loading.ProcessedDatasetConfig.model_validate(ds_cfg_dict),
            ds_path=self._cfg.test_ds_path,
            metadata_path=self._cfg.processing_metadata_path,
            batch_size=self._cfg.batch_size,
            n_workers=0,
            train_val_split=0.0,
        )

        data_module.setup(stage='test')

        return data_module
