"""Contains miscellaneous utility functions."""
import dataclasses
import hashlib
import os

import mlflow


def hash_restaurant_href(restaurant_href: str) -> str:
    """Generates a hash for a restaurant href."""

    return hashlib.sha256(bytes(restaurant_href, encoding='utf-8')).hexdigest()


@dataclasses.dataclass(unsafe_hash=True)
class MLRun:
    """Represents an MLflow run."""
    experiment_name: str
    run_name: str
    path: str


def get_mlruns(experiment_name: str):
    """Generates an MLRun object given experiment and run names."""

    experiment = mlflow.get_experiment_by_name(experiment_name)

    assert experiment is not None, f'Experiment {experiment_name} does not exist.'

    mlflow_client = mlflow.tracking.MlflowClient()

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    return [MLRun(experiment_name=experiment_name,
                  run_name=run.info.run_name,
                  path=os.path.join('mlruns', experiment.experiment_id, run.info.run_id))
            for run in runs]
