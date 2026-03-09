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
from matplotlib.axes import Axes
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

        self._save_summary_table(df, output_path / 'summary_table.csv')
        self._save_dataframe_summary(df, output_path / 'summary_stats.json')

        for metric_cfg in self._config.take_metrics:
            _logger().info('Summarizing metric: %s', metric_cfg.name)

            metric_dir = output_path / metric_cfg.name.replace('/', '-')
            metric_dir.mkdir(parents=True, exist_ok=True)

            if metric_cfg.name not in df.columns:
                _logger().warning('Metric %s not found in runs', metric_cfg.name)
                continue

            sorted_df = df.sort_values(by=metric_cfg.name, ascending=metric_cfg.mode == "min")

            self._plot_best_and_worst_curves(
                sorted_df=sorted_df,
                metric_name=metric_cfg.name,
                output_path=metric_dir / 'best_worst_curves.svg',
                n_curves=self._config.draw_n_best_curves,
                title_suffix=metric_cfg.name
            )

            self._plot_metric_distributions(
                sorted_df,
                metric_cfg.name,
                metric_dir
            )

    def get_best_runs(self, take_best_runs_by: list[MetricConfig]) -> dict[str, str]:
        """Returns best runs for selected metrics as metric-to-run mapping.

        For each selected metric, the single best run is selected according to
        the metric mode (``min``/``max``).

        Args:
            take_best_runs_by: Metric definitions used specifically to choose best runs.

        Returns:
            A dictionary mapping metrics to run identifiers that achieved the best value for
                those metrics.
        """
        df = self._create_summary_dataframe()

        best_runs: dict[str, str] = {}

        for metric_cfg in take_best_runs_by:
            metric_name = metric_cfg.name

            if metric_name not in df.columns:
                _logger().warning('Metric %s not found in runs', metric_name)
                continue

            if metric_cfg.mode == 'min':
                best_index = df[metric_name].idxmin()
            else:
                best_index = df[metric_name].idxmax()

            best_runs[metric_name] = df.loc[best_index, 'run_id']

        return best_runs

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
        """Saves the summary table to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def _save_dataframe_summary(self, df: pd.DataFrame, output_path: Path) -> None:
        """Saves dataframe summary statistics."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(df.describe(include='all').to_dict(), f, indent=4, ensure_ascii=False)

    def _plot_best_and_worst_curves(
            self,
            sorted_df: pd.DataFrame,
            metric_name: str,
            output_path: Path,
            n_curves: int,
            title_suffix: str | None = None) -> None:
        """Plots best and worst learning curves for a sorted metric dataframe."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        best_df = sorted_df.head(n_curves)
        worst_df = sorted_df.tail(n_curves)
        if best_df.empty and worst_df.empty:
            _logger().warning('No runs selected for plotting')
            return

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

        best_max_step = self._plot_learning_curves(best_df, metric_name, axes[0])
        worst_max_step = self._plot_learning_curves(worst_df, metric_name, axes[1])

        max_step = max(best_max_step, worst_max_step)
        if max_step > 0:
            axes[0].set_xlim(0, max_step)
            axes[1].set_xlim(0, max_step)

        axes[0].set_title(f'Best {n_curves}')
        axes[1].set_title(f'Worst {n_curves}')
        axes[1].set_ylabel('')

        for axis in axes:
            axis.legend()

        title_suffix = metric_name if title_suffix is None else title_suffix
        fig.suptitle(
            f'Best {n_curves} and worst {n_curves} learning curves - {title_suffix}'
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_learning_curves(
            self,
            df: pd.DataFrame,
            metric_name: str,
            axis: Axes) -> int:
        """Plots validation and train learning curves for selected runs on an axis."""

        cmap = eda_utils.get_gradient_cmap()
        sampled_colors = [
            cmap(value) for value in np.linspace(0.15, 0.95, max(len(df), 1))
        ]
        max_step = 0

        for idx, (_, row) in enumerate(df.iterrows()):
            val_steps, val_values = self._extract_metric_history(row['run_id'], metric_name)

            axis.plot(
                val_steps,
                val_values,
                linestyle='-',
                linewidth=2,
                color=sampled_colors[idx],
                label=row['run_name']
            )

            max_step = max(max_step, int(np.max(val_steps)))

            train_metric_name = metric_name.replace('val/', 'train/')

            if train_metric_name in df.columns:
                train_steps, train_values = self._extract_metric_history(
                    row['run_id'], train_metric_name)
                train_steps, train_values = self._average_over_reference_windows(
                    train_steps, train_values, val_steps)

                axis.plot(
                    train_steps,
                    train_values,
                    linestyle='--',
                    linewidth=2,
                    color=sampled_colors[idx],
                    alpha=0.7
                )

                max_step = max(max_step, int(np.max(train_steps)))

        axis.set_xlabel('Step')
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.3, axis='y')
        axis.set_axisbelow(True)

        return max_step

    def _extract_metric_history(self,
                                run_id: str,
                                metric_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Extracts metric history arrays from a run."""
        metric_history = self._mlflow_client.get_metric_history(run_id, metric_name)

        steps = np.array([m.step for m in metric_history])
        values = np.array([m.value for m in metric_history])

        return steps, values

    def _average_over_reference_windows(self,
                                        source_steps: np.ndarray,
                                        source_values: np.ndarray,
                                        reference_steps: np.ndarray
                                        ) -> tuple[np.ndarray, np.ndarray]:
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

    def _get_parameter_columns(self, df: pd.DataFrame) -> list[str]:
        """Identifies parameter columns based on common prefixes."""
        param_col_prefixes = ('model_cfg', 'train_cfg', 'ds_cfg', 'optimizer_cfg')
        return [col
                for col in df.columns
                if any(col.startswith(prefix) for prefix in param_col_prefixes)]

    def _plot_metric_distributions(
            self,
            df: pd.DataFrame,
            metric_name: str,
            output_dir: Path) -> None:
        """Plots metric distributions grouped by parameter values."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for param in self._get_parameter_columns(df):
            if df[param].nunique(dropna=False) <= 1:
                continue

            param_dir = output_dir / f'distribution_wrt_{param.replace("/", "-")}'
            param_dir.mkdir(parents=True, exist_ok=True)

            self._plot_single_param_distribution(
                df=df,
                param_name=param,
                metric_name=metric_name,
                output_path=param_dir / 'distribution.svg'
            )

            self._plot_best_and_worst_per_param_value(
                df=df,
                param_name=param,
                metric_name=metric_name,
                output_dir=param_dir
            )

    def _plot_single_param_distribution(
            self,
            df: pd.DataFrame,
            param_name: str,
            metric_name: str,
            output_path: Path) -> None:
        """Plots a metric distribution for a single parameter."""
        plot_df = df[[param_name, metric_name]].copy()
        plot_df[param_name] = plot_df[param_name].where(plot_df[param_name].notna(), 'None')
        counts = plot_df[param_name].value_counts(dropna=False)

        order = counts.index.to_series().astype(str).sort_values().index.tolist()
        label_mapping = {
            value: f'{value} (n={counts[value]})'
            for value in order
        }
        plot_df[f'{param_name}__label'] = plot_df[param_name].map(label_mapping)
        label_order = [label_mapping[value] for value in order]

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.violinplot(
            data=plot_df,
            x=f'{param_name}__label',
            y=metric_name,
            hue=f'{param_name}__label',
            inner='quart',
            order=label_order,
            palette=eda_utils.get_gradient_palette_reversed(len(label_order)),
            legend=False,
            ax=ax
        )

        ax.set_xlabel(param_name.rsplit('/', maxsplit=1)[-1] or param_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Distribution of "{metric_name}" by "{param_name}"')
        ax.xaxis.grid(True, 'minor', linewidth=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)

    def _plot_best_and_worst_per_param_value(
            self,
            df: pd.DataFrame,
            param_name: str,
            metric_name: str,
            output_dir: Path) -> None:
        """Plots single best and worst curves for each value of a parameter."""

        values = df[param_name].drop_duplicates().tolist()

        for value in values:
            if pd.isna(value):
                subset_df = df[df[param_name].isna()].copy()
                value_name = 'None'
            else:
                subset_df = df[df[param_name] == value].copy()
                value_name = str(value)

            output_path = output_dir / f'curves_{self._sanitize_name(value_name)}.svg'
            self._plot_best_and_worst_curves(
                sorted_df=subset_df,
                metric_name=metric_name,
                output_path=output_path,
                n_curves=1,
                title_suffix=f'{param_name}={value_name}'
            )

    def _sanitize_name(self, value: str) -> str:
        """Sanitizes a string for usage in file names."""
        return value.replace('/', '-').replace(' ', '_')
