"""Shared utilities for experiment summarization workflows."""
from pathlib import Path
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel


class MetricConfig(BaseModel):
    """Configuration for a metric to summarize."""
    name: str
    mode: Literal['min', 'max']


def create_summary_dataframe(
        mlflow_client: mlflow.tracking.MlflowClient,
        run_ids: list[str]) -> pd.DataFrame:
    """Builds a summary dataframe for explicitly selected run ids."""

    unique_run_ids = list(dict.fromkeys(run_ids))
    runs = [mlflow_client.get_run(run_id) for run_id in unique_run_ids]

    data = [
        {
            'run_name': run.data.tags.get('mlflow.runName', run.info.run_id),
            'run_id': run.info.run_id,
            'experiment_name': get_experiment_name(mlflow_client, run.info.experiment_id),
            **run.data.metrics,
            **run.data.params,
        }
        for run in runs
    ]

    df = pd.DataFrame(data)

    for column in get_parameter_columns(df):

        numeric_values = pd.to_numeric(df[column], errors='coerce')
        if numeric_values.notna().sum() == df[column].notna().sum():
            df[column] = numeric_values

    return df


def get_experiment_run_ids(
        mlflow_client: mlflow.tracking.MlflowClient,
        experiment_name: str) -> list[str]:
    """Returns run identifiers for a given experiment name."""
    experiment = mlflow_client.get_experiment_by_name(experiment_name)

    if not experiment:
        return []

    runs = mlflow_client.search_runs(experiment.experiment_id)
    return [run.info.run_id for run in runs]


def save_summary_table(df: pd.DataFrame, output_path: Path) -> None:
    """Saves the summary table to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def extract_metric_history(
        mlflow_client: mlflow.tracking.MlflowClient,
        run_id: str,
        metric_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Extracts metric history arrays from a run."""
    metric_history = mlflow_client.get_metric_history(run_id, metric_name)

    steps = np.array([m.step for m in metric_history])
    values = np.array([m.value for m in metric_history])

    return steps, values


def average_over_reference_windows(
        source_steps: np.ndarray,
        source_values: np.ndarray,
        reference_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Averages source metric values in windows defined by reference steps."""
    if len(source_steps) == 0 or len(reference_steps) == 0:
        return np.array([]), np.array([])

    averaged_steps = []
    averaged_values = []
    previous_step = -np.inf

    for step in reference_steps:
        window = (source_steps > previous_step) & (source_steps <= step)

        if np.any(window):
            averaged_steps.append(step)
            averaged_values.append(np.mean(source_values[window]))

        previous_step = step

    return np.array(averaged_steps), np.array(averaged_values)


def get_parameter_columns(df: pd.DataFrame) -> list[str]:
    """Identifies parameter columns based on common prefixes."""
    param_col_prefixes = ('model_cfg', 'train_cfg', 'ds_cfg', 'optimizer_cfg')
    return [col
            for col in df.columns
            if any(col.startswith(prefix) for prefix in param_col_prefixes)]


def sanitize_name(value: str) -> str:
    """Sanitizes a string for usage in file names."""
    return value.replace('/', '-').replace(' ', '_')


def sanitize_metric_name(metric_name: str) -> str:
    """Sanitizes metric names for file names."""
    return metric_name.replace('/', '-')


def get_experiment_name(
        mlflow_client: mlflow.tracking.MlflowClient,
        experiment_id: str) -> str:
    """Returns experiment name for a given experiment id."""
    return mlflow_client.get_experiment(experiment_id).name  # type: ignore
