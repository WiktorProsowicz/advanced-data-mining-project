"""Utilities for summarizing globally selected best MLflow runs."""
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class BestRunsSummarizer:
    """Builds a global summary for best runs selected per metric."""

    def __init__(self,
                 mlflow_client: mlflow.tracking.MlflowClient,
                 parameter_names: list[str]):
        """Initializes the summarizer.

        Args:
            mlflow_client: MLflow client used to fetch run metadata.
            parameter_names: Parameter names for metric-vs-parameter scatter plots.
        """
        self._mlflow_client = mlflow_client
        self._parameter_names = parameter_names

    def summarize(self,
                  best_runs_by_metric: dict[str, list[str]],
                  output_path: Path) -> None:
        """Writes per-metric summary tables and scatter plots.

        Args:
            best_runs_by_metric: Mapping from metric name to list of run IDs selected
                as best for that metric.
            output_path: Base output directory.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        for metric_name, run_ids in best_runs_by_metric.items():
            metric_dir = output_path / self._sanitize_metric_name(metric_name)
            metric_dir.mkdir(parents=True, exist_ok=True)

            summary_df = self._create_summary_dataframe(run_ids)
            summary_df.to_csv(metric_dir / 'summary_table.csv', index=False)

            self._save_metric_parameter_scatter_plots(
                metric_name=metric_name,
                summary_df=summary_df,
                parameter_names=self._parameter_names,
                metric_dir=metric_dir,
            )

    def _create_summary_dataframe(self,
                                  run_ids: list[str]) -> pd.DataFrame:
        """Creates a table with run names, metrics, and parameters."""

        unique_run_ids = list(dict.fromkeys(run_ids))

        rows: list[dict[str, object]] = []
        for run_id in unique_run_ids:
            run = self._mlflow_client.get_run(run_id)
            row: dict[str, object] = {
                'run_name': run.data.tags.get('mlflow.runName', run.info.run_id),
                'run_id': run.info.run_id,
                'experiment_name': self._get_experiment_name(run.info.experiment_id),
            }
            row.update(run.data.metrics)
            row.update(dict(run.data.params))

            rows.append(row)

        df = pd.DataFrame(rows)

        for column in self._get_parameter_columns(df):
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            if numeric_values.notna().sum() == df[column].notna().sum():
                df[column] = numeric_values

        return df

    def _get_parameter_columns(self, df: pd.DataFrame) -> list[str]:
        """Identifies parameter columns based on common prefixes."""
        param_col_prefixes = ('model_cfg', 'train_cfg', 'ds_cfg', 'optimizer_cfg')
        return [col
                for col in df.columns
                if any(col.startswith(prefix) for prefix in param_col_prefixes)]

    def _save_metric_parameter_scatter_plots(self,
                                             metric_name: str,
                                             summary_df: pd.DataFrame,
                                             parameter_names: list[str],
                                             metric_dir: Path) -> None:
        """Saves scatter plots for metric values with respect to parameter values."""

        if metric_name not in summary_df.columns:
            return

        plot_df = summary_df.copy()
        plot_df[metric_name] = pd.to_numeric(plot_df[metric_name], errors='coerce')
        plot_df = plot_df[plot_df[metric_name].notna()]

        if plot_df.empty:
            return

        for parameter_name in parameter_names:
            if parameter_name not in plot_df.columns:
                _logger().warning('Parameter "%s" not found in summary table for metric "%s".',
                                  parameter_name, metric_name)
                continue

            param_metric_df = plot_df[[parameter_name, metric_name]].dropna()

            fig, axis = plt.subplots(figsize=(12, 6))
            axis.scatter(param_metric_df[parameter_name], param_metric_df[metric_name])
            axis.set_xlabel(parameter_name)
            axis.set_ylabel(metric_name)
            axis.set_title(f'{metric_name} vs {parameter_name}')
            axis.tick_params(axis='x', rotation=45)
            axis.grid(True, axis='y', alpha=0.3)
            axis.set_axisbelow(True)

            fig.tight_layout()
            fig.savefig(
                metric_dir / f'scatter_wrt_{self._sanitize_metric_name(parameter_name)}.svg',
                dpi=150)
            plt.close(fig)

    def _sanitize_metric_name(self, metric_name: str) -> str:
        """Sanitizes metric names for file names."""
        return metric_name.replace('/', '-')

    def _get_experiment_name(self, experiment_id: str) -> str:
        """Returns experiment name for a given experiment id."""
        return self._mlflow_client.get_experiment(experiment_id).name  # type: ignore
