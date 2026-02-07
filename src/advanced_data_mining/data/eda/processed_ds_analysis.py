"""Module that constructs stats and visualizations for processed datasets."""

import pathlib
import json
from typing import Any, Iterator

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm

from advanced_data_mining.data.eda import utils as eda_utils
from advanced_data_mining.data.structs import processed_ds
from advanced_data_mining.data import ds_loading


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

        for trace_spec in trace_features_df['trace_spec'].unique():

            sub_df = trace_features_df[trace_features_df['trace_spec'] == trace_spec]

            self._save_trace_features_distributions(
                sub_df,
                hue='Is Translated',
                cmap=eda_utils.get_gradient_palette(2),
                output_path=output_dir / f'{trace_spec}_by_translated.svg'
            )

            self._save_trace_features_distributions(
                sub_df,
                hue='Rating',
                cmap=eda_utils.get_coolwarm_cmap(),
                output_path=output_dir / f'{trace_spec}_by_rating.svg'
            )

            self._save_trace_features_distributions(
                sub_df,
                hue='Number of sentences',
                cmap=eda_utils.get_gradient_cmap(),
                output_path=output_dir / f'{trace_spec}_by_n_sentences.svg'
            )

            self._save_trace_features_distributions(
                sub_df,
                hue='Number of words',
                cmap=eda_utils.get_gradient_cmap(),
                output_path=output_dir / f'{trace_spec}_by_n_words.svg'
            )

            general_stats[trace_spec] = (
                sub_df[['trace_velocity', 'trace_volume']]
                .describe(percentiles=[0.01, .05, .25, .5, .75, .9, .95, .99],).to_dict()
            )

            general_stats[trace_spec]['n_zeros'] = sub_df[sub_df['trace_volume'] == 0.0].shape[0]

        with (output_dir / 'general_stats.json').open('w', encoding='utf-8') as f:
            json.dump(general_stats, f, ensure_ascii=False, indent=4)

    def _generate_trace_features_df(self) -> Iterator[dict[str, Any]]:
        """Generates a DataFrame containing trace features for all reviews."""

        ds = ds_loading.ProcessedDataset(
            ds_loading.ProcessedDatasetConfig(
                use_trace_features=None,
                use_categorized_features=[]
            ),
            metadata_handler=self._metadata_path_handler,
            samples=[
                review for restaurant in self._ds_path_handler.iter_restaurants()
                for review in self._ds_path_handler.iter_reviews_for(restaurant)
            ]
        )

        for sample_idx in tqdm.tqdm(range(len(ds)),
                                    desc='Generating trace features dataframe',
                                    unit='samples'):
            sample_data = ds[sample_idx]
            sample_metadata = ds.get_raw_sample(sample_idx)

            for key in sample_data:
                if key.startswith('trace_'):
                    yield {
                        'trace_spec': key,
                        'trace_velocity': float(sample_data[key][0]),
                        'trace_volume': float(sample_data[key][1]),
                        'Rating': int(sample_data['rating']),
                        'Is Translated': sample_metadata.is_translated,
                        'Number of words': sample_metadata.n_words,
                        'Number of sentences': sample_metadata.n_sentences
                    }

    def _generate_numerical_features_df(self) -> Iterator[dict[str, Any]]:
        """Generates a DataFrame containing numerical features for all reviews."""

        ds = ds_loading.ProcessedDataset(
            ds_loading.ProcessedDatasetConfig(
                use_trace_features=[],
                use_categorized_features=None
            ),
            metadata_handler=self._metadata_path_handler,
            samples=[
                review for restaurant in self._ds_path_handler.iter_restaurants()
                for review in self._ds_path_handler.iter_reviews_for(restaurant)
            ]
        )

        for sample_idx in tqdm.tqdm(range(len(ds)),
                                    desc='Generating numerical features dataframe',
                                    unit='samples'):  # pylint: disable=consider-using-enumerate
            sample_data = ds[sample_idx]
            sample_metadata = ds.get_raw_sample(sample_idx)

            yield {
                **{name: int(feature) for name, feature in sample_data.items()},
                'n_author_reviews_index': sample_metadata.n_author_reviews_index,
                'is_translated': sample_metadata.is_translated
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
                                           hue: str | pd.Series,
                                           cmap: Any,
                                           output_path: pathlib.Path) -> None:
        """Saves the distributions of trace features to the output directory."""

        fig, ax = plt.subplots(figsize=(10, 10), sharey=True)

        mean_velocity = trace_features_df['trace_velocity'].mean()
        mean_volume = trace_features_df['trace_volume'].mean()

        sns.scatterplot(
            data=trace_features_df.sample(1000, random_state=42),
            x='trace_velocity',
            y='trace_volume',
            hue=hue,
            alpha=0.8,
            palette=cmap,
            ax=ax
        )

        ax.axvline(mean_velocity, color=eda_utils.LIGHT_COLOR_STD,
                   linestyle='--', label='Mean Velocity', alpha=0.3)
        ax.axhline(mean_volume, color=eda_utils.DARK_COLOR_STD,
                   linestyle='--', label='Mean Volume', alpha=0.3)
        ax.legend()

        ax.set_title(f'Trace features distribution (by \'{hue}\' feature)')
        ax.set_xlabel('Trace Velocity')
        ax.set_ylabel('Trace Volume')
        ax.grid(True, linestyle='--')
        ax.set_axisbelow(True)

        fig.savefig(output_path)
