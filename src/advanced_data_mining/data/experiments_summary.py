"""Contains utilities for summarizing MLflow experiments."""
import json
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

from advanced_data_mining.data.eda import utils as eda_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class MetricConfig(BaseModel):
    """Configuration for a metric to summarize."""
    name: str
    mode: Literal["min", "max"]


class ExperimentSummarizerConfig(BaseModel):
    """Configuration for ExperimentSummarizer."""
    experiment_name: str
    draw_n_best_curves: int
    draw_n_worst_curves: int
    take_metrics: list[MetricConfig]


class ExperimentSummarizer:
    """Summarizes MLflow experiments with metrics, plots, and figures."""

    def __init__(self, config: ExperimentSummarizerConfig,
                 mlflow_client: mlflow.tracking.MlflowClient):
        """Initializes the summarizer from configuration."""
        self._config = config
        self._mlflow_client = mlflow_client
        sns.set_theme(style='darkgrid')

    def summarize(self, output_path: Path) -> None:
        """Produces experiment summaries for each configured metric.

        The method prepares output directories, ranks runs for each metric,
        saves tabular summaries, and generates plots that highlight both
        learning behavior and distribution patterns across parameter values.

        Args:
            output_path: Directory where summary artifacts are written.
        """
        _logger().info('Starting experiment summarization for experiment: %s',
                       self._config.experiment_name)

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        df = self._create_summary_dataframe()
        if df.empty:
            _logger().warning('No runs found')
            return

        for metric_cfg in self._config.take_metrics:
            _logger().info('Summarizing metric: %s', metric_cfg.name)

            metric_dir = output_path / metric_cfg.name.replace('/', '-')
            metric_dir.mkdir(parents=True, exist_ok=True)

            if metric_cfg.name not in df.columns:
                _logger().warning('Metric %s not found in runs', metric_cfg.name)
                continue

            sorted_df = df.sort_values(by=metric_cfg.name, ascending=metric_cfg.mode == "min")

            self._save_summary_table(sorted_df, metric_dir / 'summary_table.md')

            self._save_dataframe_summary(sorted_df, metric_dir / 'summary_stats.json')

            self._plot_learning_curves(
                sorted_df, metric_cfg.name,
                self._config.draw_n_best_curves, metric_dir / 'best_curves.png', best=True
            )

            self._plot_learning_curves(
                sorted_df, metric_cfg.name,
                self._config.draw_n_worst_curves, metric_dir / 'worst_curves.png', best=False
            )

            self._plot_metric_distributions(sorted_df, metric_cfg.name, metric_dir)

    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Builds a summary dataframe for all runs."""
        data = []
        experiment = self._mlflow_client.get_experiment_by_name(
            self._config.experiment_name
        )

        if not experiment:
            _logger().warning('Experiment not found: %s', self._config.experiment_name)
            return pd.DataFrame()

        runs = self._mlflow_client.search_runs(experiment.experiment_id)

        for run in runs:
            row = {'run_name': run.data.tags.get('mlflow.runName', run.info.run_id),
                   'run_id': run.info.run_id}

            row.update(run.data.metrics)
            row.update(dict(run.data.params))

            data.append(row)

        df = pd.DataFrame(data)

        for column in self._get_parameter_columns(df):

            numeric_values = pd.to_numeric(df[column], errors='coerce')
            if numeric_values.notna().sum() == df[column].notna().sum():
                df[column] = numeric_values

        return df

    def _save_summary_table(self, df: pd.DataFrame, output_path: Path) -> None:
        """Saves the summary table to Markdown."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(df.to_markdown(index=False), encoding='utf-8')

    def _save_dataframe_summary(self, df: pd.DataFrame, output_path: Path) -> None:
        """Saves dataframe summary statistics."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(df.describe(include='all').to_dict(), f, indent=4, ensure_ascii=False)

    def _plot_learning_curves(self,
                              df: pd.DataFrame,
                              metric_name: str,
                              n_curves: int,
                              output_path: Path,
                              best: bool = True) -> None:
        """Plots learning curves for selected runs."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if best:
            selected_df = df.head(n_curves)
            title_prefix = "Best"
        else:
            selected_df = df.tail(n_curves)
            title_prefix = "Worst"

        if selected_df.empty:
            _logger().warning('No runs selected for plotting')
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        for _, row in selected_df.iterrows():
            epochs, values = self._extract_metric_history(row['run_id'], metric_name)
            ax.plot(epochs, values, marker='o', label=row['run_name'], linewidth=2)

        ax.set_xlabel('Step')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{title_prefix} {n_curves} Learning Curves - {metric_name}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _extract_metric_history(self,
                                run_id: str,
                                metric_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extracts metric history arrays from a run."""
        metric_history = self._mlflow_client.get_metric_history(run_id, metric_name)

        steps = np.array([m.step for m in metric_history])
        values = np.array([m.value for m in metric_history])

        return steps, values

    def _get_parameter_columns(self, df: pd.DataFrame) -> list[str]:
        """Identifies parameter columns based on common prefixes."""
        param_col_prefixes = ('model_cfg', 'data_cfg', 'train_cfg', 'ds_cfg')
        return [col
                for col in df.columns
                if any(col.startswith(prefix) for prefix in param_col_prefixes)]

    def _plot_metric_distributions(
            self, df: pd.DataFrame, metric_name: str, output_dir: Path) -> None:
        """Plots metric distributions grouped by parameter values."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for param in self._get_parameter_columns(df):

            fig, ax = plt.subplots(figsize=(12, 6))

            sns.violinplot(
                data=df,
                x=param,
                y=metric_name,
                hue=param,
                inner='quart',
                palette=eda_utils.get_gradient_palette_reversed(df[param].nunique()),
                legend=False,
                ax=ax
            )

            ax.set_xlabel(param.rsplit('/', maxsplit=1)[-1] or param)
            ax.set_ylabel(metric_name)
            ax.set_title(f'Distribution of "{metric_name}" by "{param}"')
            ax.xaxis.grid(True, 'minor', linewidth=0.25)
            ax.set_axisbelow(True)
            ax.tick_params(axis='x', rotation=45)

            fig.tight_layout()
            output_file = output_dir / f'distribution_by_{param.replace("/", "-")}.png'
            fig.savefig(output_file, dpi=150)
            plt.close(fig)
