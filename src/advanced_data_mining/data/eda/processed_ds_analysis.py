"""Module that constructs stats and visualizations for processed datasets."""

import pathlib
import json
from typing import Iterator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from advanced_data_mining.data.eda import utils as eda_utils
from advanced_data_mining.data.structs import processed_ds


class ProcessedDatasetAnalyzer:
    """Composes statistics and visualizations for processed datasets."""

    def __init__(self,
                 processed_ds_path: pathlib.Path,
                 processing_metadata_path: pathlib.Path):

        self._ds_path_handler = processed_ds.ProcessedDsPathHandler(processed_ds_path)
        self._metadata_path_handler = processed_ds.ProcessingMetadataPathHandler(
            processing_metadata_path)

    def save_numerical_feature_distributions(self, output_dir: pathlib.Path) -> None:
        """Saves visualizations of numerical feature distributions to the output directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        features_df = pd.DataFrame(self._generate_numerical_features_df())

        for normalize in [True, False]:

            base_out_dir = output_dir / ('normalized' if normalize else 'absolute')
            base_out_dir.mkdir(parents=True, exist_ok=True)

            for feature_name in self._metadata_path_handler.get_supported_cat_opt_names():

                self._save_cat_feature_distribution(
                    features_df,
                    feature_name,
                    self._metadata_path_handler.get_cat_opt_label_mapping(feature_name),
                    normalize_to_one=normalize,
                    output_path=base_out_dir / f'{feature_name}_distribution.svg')

            self._save_cat_feature_distribution(
                features_df,
                'n_author_reviews_index',
                self._metadata_path_handler.get_n_author_reviews_label_mapping(),
                normalize_to_one=normalize,
                output_path=base_out_dir / 'n_author_reviews_index_distribution.svg')

    def save_trace_features_stats(self, output_dir: pathlib.Path) -> None:
        """Saves statistics and distributions about trace features to the output path."""

        output_dir.mkdir(parents=True, exist_ok=True)

        trace_features_df = pd.DataFrame(self._generate_trace_features_df())

        general_stats = {}

        for chunk_size, step_size in self._metadata_path_handler.get_chunk_and_step_sizes():

            sub_df = trace_features_df[
                (trace_features_df['chunk_length'] == chunk_size) &
                (trace_features_df['step_size'] == step_size)
            ]

            self._save_trace_features_distributions(
                sub_df,
                output_path=output_dir / f'trace_features_chunk_{chunk_size}_step_{step_size}.svg'
            )

            general_stats[f'chunk_{chunk_size}_step_{step_size}'] = sub_df.describe().to_dict()

        with (output_dir / 'general_stats.json').open('w', encoding='utf-8') as f:
            json.dump(general_stats, f, ensure_ascii=False, indent=4)

    def _generate_trace_features_df(self) -> Iterator[dict[str, str]]:
        """Generates a DataFrame containing trace features for all reviews."""

        for restaurant in self._ds_path_handler.iter_restaurants():
            for review in self._ds_path_handler.iter_reviews_for(restaurant):
                with review.trace_features_pth.open('r', encoding='utf-8') as f:
                    trace_features = json.load(f)

                with review.num_features_pth.open('r', encoding='utf-8') as f:
                    num_features = json.load(f)

                for feature in trace_features:
                    yield {
                        'chunk_length': feature['chunk_length'],
                        'step_size': feature['step_size'],
                        'trace_velocity': feature['trace_velocity'],
                        'trace_volume': feature['trace_volume'],
                        'is_translated': num_features['is_translated'],
                        'rating': num_features['rating']
                    }

    def _generate_numerical_features_df(self) -> Iterator[dict[str, str]]:
        """Generates a DataFrame containing numerical features for all reviews."""

        for restaurant in self._ds_path_handler.iter_restaurants():
            for review in self._ds_path_handler.iter_reviews_for(restaurant):
                with review.num_features_pth.open('r', encoding='utf-8') as f:
                    num_features = json.load(f)

                yield {
                    **num_features['encoded_cat_options'],
                    'n_author_reviews_index': num_features['n_author_reviews_index'],
                    'is_translated': num_features['is_translated'],
                    'rating': num_features['rating']
                }

    def _save_cat_feature_distribution(self,
                                       features_df: pd.DataFrame,
                                       feature_name: str,
                                       label_mapping: dict[int, str],
                                       normalize_to_one: bool,
                                       output_path: pathlib.Path) -> None:
        """Saves the distribution of a categorical feature to the output directory."""

        df = features_df.copy()
        df['share'] = 1.0

        if normalize_to_one:
            df.loc[df['is_translated'], 'share'] /= df[df['is_translated']]['share'].sum()
            df.loc[~df['is_translated'], 'share'] /= df[~df['is_translated']]['share'].sum()

        df = pd.concat([df, pd.DataFrame([{feature_name: key, 'share': 0.0}
                                          for key in label_mapping.keys()])])

        graph = sns.catplot(
            data=df,
            x=feature_name,
            y='share',
            hue='is_translated',
            kind='bar',
            estimator=sum,
            height=6,
            aspect=1.5,
            alpha=.8,
            palette=eda_utils.get_gradient_palette(2),
            legend=True
        )

        graph.ax.set_title(f'Distribution of categorical feature: "{feature_name}"')
        graph.ax.set_xlabel(feature_name)
        graph.ax.set_ylabel('Share')
        graph.ax.grid(True, axis='y', linestyle='--')
        graph.ax.set_axisbelow(True)
        graph.ax.set_xticks([i for i, _ in enumerate(label_mapping)],
                            [label_mapping[i] for i in range(len(label_mapping))])

        if normalize_to_one:
            graph.ax.set_ylim(0, 1)

        graph.savefig(output_path)

    def _save_trace_features_distributions(self,
                                           trace_features_df: pd.DataFrame,
                                           output_path: pathlib.Path) -> None:
        """Saves the distributions of trace features to the output directory."""

        fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

        sns.scatterplot(
            data=(trace_features_df
                  .groupby('is_translated')
                  .apply(lambda x: x.sample(1000, random_state=42), include_groups=False)),
            x='trace_velocity',
            y='trace_volume',
            hue='is_translated',
            alpha=0.6,
            palette=eda_utils.get_gradient_palette(2),
            ax=ax[0]
        )

        sns.scatterplot(
            data=(trace_features_df
                  .groupby('rating')
                  .apply(lambda x: x.sample(400, random_state=42), include_groups=False)),
            x='trace_velocity',
            y='trace_volume',
            hue='rating',
            alpha=0.6,
            palette=eda_utils.get_coolwarm_cmap(),
            ax=ax[1]
        )

        ax[0].set_title('Trace Features by Translation Status')
        ax[0].set_xlabel('Trace Velocity')
        ax[0].set_ylabel('Trace Volume')
        ax[0].grid(True, linestyle='--')
        ax[0].set_axisbelow(True)

        ax[1].set_title('Trace Features by Rating')
        ax[1].set_xlabel('Trace Velocity')
        ax[1].set_ylabel('Trace Volume')
        ax[1].grid(True, linestyle='--')
        ax[1].set_axisbelow(True)

        fig.savefig(output_path)
