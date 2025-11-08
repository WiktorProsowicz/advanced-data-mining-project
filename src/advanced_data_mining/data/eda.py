# -*- coding: utf-8 -*-
"""Contains utilities for exploratory data analysis (EDA)."""
import os
import re
from typing import Any
from typing import Dict
from typing import Set
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch

from advanced_data_mining.utils import misc


def is_outlier(series: pd.Series) -> bool:
    """Determines what sentences are outliers using the IQR method."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)


class EDAFeatureExtractor:
    """Extracts features from the dataset for EDA purposes."""

    def __init__(self, processed_ds_path: str):

        self._processed_ds_path = processed_ds_path

        self._ds = pd.read_pickle(os.path.join(self._processed_ds_path, 'preprocessed_dataset.pkl'))

        self._paths_df = self._ds[['restaurant_href']]

        for col in ['bow_representations_bottom',
                    'bow_representations_top',
                    'bow_representations_full',
                    'tfidf_representations_bottom',
                    'tfidf_representations_top',
                    'tfidf_representations_full',
                    'sentence_bert_embeddings']:

            paths_series = self._ds.index.to_series().apply(
                lambda idx: os.path.join(self._processed_ds_path,
                                         col,  # pylint: disable=cell-var-from-loop
                                         misc.hash_restaurant_href(
                                             self._ds.at[idx, 'restaurant_href']),
                                         f'{idx}.pt')
            )

            self._paths_df = pd.concat([self._paths_df, paths_series.rename(col)], axis=1)

        self._numerical_stats = pd.read_pickle(os.path.join(self._processed_ds_path,
                                                            'numerical_features.pkl'))

        self._bow_nzero_df = self._get_bbow_nzero_df()

    def extract_basic_stats(self) -> Dict[str, Any]:
        """Extracts basic statistics from the dataset."""

        stats: Dict[str, Any] = {}

        stats['total_restaurants'] = int(self._ds['restaurant_href'].nunique())
        stats['total_reviews'] = int(len(self._ds))
        stats['avg_review_per_restaurant'] = float(
            stats['total_reviews'] / stats['total_restaurants'])

        restaurant_ratings = self._ds.groupby('restaurant_href')['review_rating'].mean()

        stats['avg_restaurant_rating'] = float(restaurant_ratings.mean())
        stats['min_restaurant_rating'] = float(restaurant_ratings.min())
        stats['max_restaurant_rating'] = float(restaurant_ratings.max())

        stats['avg_words_in_review'] = float(self._numerical_stats['num_words'].mean())
        stats['min_words_in_review'] = float(self._numerical_stats['num_words'].min())
        stats['max_words_in_review'] = float(self._numerical_stats['num_words'].max())

        stats['avg_sentences_in_review'] = float(self._numerical_stats['num_sentences'].mean())
        stats['min_sentences_in_review'] = float(self._numerical_stats['num_sentences'].min())
        stats['max_sentences_in_review'] = float(self._numerical_stats['num_sentences'].max())

        stats['n_reviews_from_cracow'] = int(len(self._ds[self._ds['is_from_cracow']]))

        for chunk_length, chunk_size in self._get_vol_vel_chunk_infos():

            vel_vol_stats: Dict[str, float] = {}

            vel_col = f'trace_velocity_cl_{chunk_length}_sz_{chunk_size}'
            vol_col = f'trace_volume_cl_{chunk_length}_sz_{chunk_size}'

            vel_vol_stats['min_velocity'] = float(self._numerical_stats[vel_col].min())
            vel_vol_stats['max_velocity'] = float(self._numerical_stats[vel_col].max())
            vel_vol_stats['avg_velocity'] = float(self._numerical_stats[vel_col].mean())

            vel_vol_stats['min_volume'] = float(self._numerical_stats[vol_col].min())
            vel_vol_stats['max_volume'] = float(self._numerical_stats[vol_col].max())
            vel_vol_stats['avg_volume'] = float(self._numerical_stats[vol_col].mean())

            vel_vol_stats['n_reviews_with_zero_velocity'] = int(
                len(self._numerical_stats[self._numerical_stats[vel_col] == 0]))
            vel_vol_stats['n_reviews_with_zero_volume'] = int(
                len(self._numerical_stats[self._numerical_stats[vol_col] == 0]))

            stats[f'velocity_volume_stats_cl_{chunk_length}_sz_{chunk_size}'] = vel_vol_stats

        bbow_stats: Dict[str, Any] = {}

        for bbow_type in ['top', 'bottom', 'full']:

            bbow_stats[bbow_type] = {
                'min_non_zero': int(self._bow_nzero_df[bbow_type].min()),
                'max_non_zero': int(self._bow_nzero_df[bbow_type].max()),
                'avg_non_zero': float(self._bow_nzero_df[bbow_type].mean())
            }

        stats['bbow_stats'] = bbow_stats

        traslation_stats: Dict[str, Any] = {}

        traslation_stats['n_reviews_translated'] = int(
            len(self._ds[self._ds['is_translated']]))

        translated_percentages = self._ds.groupby('review_rating')['is_translated'].mean()

        for rating in range(1, 6):
            traslation_stats[f'rating_{rating}_translated_percentage'] = float(
                translated_percentages[rating]
            )

        stats['translation_stats'] = traslation_stats

        return stats

    def get_figures(self) -> Dict[str, plt.Figure]:
        """Generates figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        figures.update(self._get_distribution_figures())

        figures.update(self._get_velocity_volume_figures())

        figures.update(self._get_length_clustering_figures())

        return figures

    def get_example_reviews(self) -> Dict[str, Any]:
        """Returns example reviews from the dataset.

        The reviews are selected from those containing most/least frequent words, as well as the
        outliers in terms of length.
        """

        sentence_outliers = self._ds[
            is_outlier(self._numerical_stats['num_sentences'])
        ].sample(n=5)

        word_outliers = self._ds[
            is_outlier(self._numerical_stats['num_words'])
        ].sample(n=5)

        containing_most_frequent = self._ds[
            self._bow_nzero_df['top'] >= 0
        ].sample(n=5)

        containing_least_frequent = self._ds[
            self._bow_nzero_df['bottom'] >= 0
        ].sample(n=5)

        return {
            'sentence_outliers': sentence_outliers.to_dict(orient='records'),
            'word_outliers': word_outliers.to_dict(orient='records'),
            'containing_most_frequent': containing_most_frequent.to_dict(orient='records'),
            'containing_least_frequent': containing_least_frequent.to_dict(orient='records'),
        }

    def _get_length_clustering_figures(self) -> Dict[str, plt.Figure]:
        """Generates figures showing clustering with respect to review length."""

        fig, axes = plt.subplots(2, figsize=(8, 12))

        chosen_reviews = self._numerical_stats.sample(n=min(1000, len(self._numerical_stats)))

        for rating in range(1, 6):
            axes[0].scatter(
                chosen_reviews['num_words'][chosen_reviews['review_rating'] == rating],
                chosen_reviews['num_sentences'][chosen_reviews['review_rating'] == rating],
                alpha=0.5,
                label=f'Rating {rating}'
            )

        axes[0].set_title('Clustering of Number of Words vs Number of Sentences')
        axes[0].set_xlabel('Number of Words')
        axes[0].set_ylabel('Number of Sentences')
        axes[0].legend()

        axes[1].scatter(chosen_reviews['num_words'][chosen_reviews['is_from_cracow']],
                        chosen_reviews['num_sentences'][chosen_reviews['is_from_cracow']],
                        alpha=0.5, label='From Cracow')
        axes[1].scatter(chosen_reviews['num_words'][~chosen_reviews['is_from_cracow']],
                        chosen_reviews['num_sentences'][~chosen_reviews['is_from_cracow']],
                        alpha=0.5, label='From Warsaw')
        axes[1].set_title('Clustering of Number of Words vs Number of Sentences')
        axes[1].set_xlabel('Number of Words')
        axes[1].set_ylabel('Number of Sentences')
        axes[1].legend()

        return {
            'clustering_words_vs_sentences': fig
        }

    def _get_distribution_figures(self) -> Dict[str, plt.Figure]:
        """Generates distribution figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        fig, ax = plt.subplots()

        ax.hist(self._ds['review_rating'], bins=5, range=(1, 6),
                align='left', rwidth=0.8,
                weights=np.ones(len(self._ds['review_rating'])) / len(self._ds['review_rating']))
        ax.set_title('Distribution of Review Ratings')
        ax.set_xlabel('Review Rating')
        ax.set_ylabel('Percentage of Reviews')
        ax.yaxis.set_major_formatter(plticker.PercentFormatter(1))

        figures['review_rating_distribution'] = fig

        threshold = self._numerical_stats['num_words'].quantile(0.95)
        chosen_reviews = self._numerical_stats[self._numerical_stats['num_words'] < threshold]

        fig, axes = plt.subplots(5, figsize=(8, 25))

        for i, rating in enumerate(range(1, 6)):
            ratings = chosen_reviews[chosen_reviews['review_rating'] == rating]
            axes[i].hist(ratings['num_words'], bins=100, color='orange')
            axes[i].set_title(f'Distribution of Number of Words in Reviews (Rating {rating})')
            axes[i].set_xlabel('Number of Words')
            axes[i].set_ylabel('Number of Reviews')

        figures['num_words_distribution'] = fig

        threshold = self._numerical_stats['num_sentences'].quantile(0.95)
        chosen_reviews = self._numerical_stats[self._numerical_stats['num_sentences'] < threshold]

        fig, axes = plt.subplots(5, figsize=(8, 25))

        for i, rating in enumerate(range(1, 6)):
            ratings = chosen_reviews[chosen_reviews['review_rating'] == rating]
            axes[i].hist(ratings['num_sentences'], bins=100, color='green')
            axes[i].set_title(f'Distribution of Number of Sentences in Reviews (Rating {rating})')
            axes[i].set_xlabel('Number of Sentences')
            axes[i].set_ylabel('Number of Reviews')

        figures['num_sentences_distribution'] = fig

        fig, axes = plt.subplots(3, figsize=(8, 18))

        for i, bow_type in enumerate(['top', 'bottom', 'full']):
            axes[i].hist(self._bow_nzero_df[bow_type], bins=100, color='purple')
            axes[i].set_title(
                f'Distribution of Non-Zero BOW Features ({bow_type.capitalize()} Words)')
            axes[i].set_xlabel('Number of Non-Zero BOW Features')
            axes[i].set_ylabel('Number of Reviews')

        figures['bow_nzero_distribution'] = fig

        return figures

    def _get_velocity_volume_figures(self) -> Dict[str, plt.Figure]:
        """Generates velocity and volume clustering figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        chunk_infos = self._get_vol_vel_chunk_infos()

        fig, axes = plt.subplots(len(chunk_infos), 3, figsize=(12, 6 * len(chunk_infos)))

        for i, (chunk_length, chunk_size) in enumerate(chunk_infos):

            vel_col = f'trace_velocity_cl_{chunk_length}_sz_{chunk_size}'
            vol_col = f'trace_volume_cl_{chunk_length}_sz_{chunk_size}'

            chosen_reviews = pd.concat([self._numerical_stats[vel_col],
                                        self._numerical_stats[vol_col],
                                        self._numerical_stats['review_rating'],
                                        self._numerical_stats['is_from_cracow'],
                                        self._ds['is_translated']], axis=1)
            chosen_reviews = chosen_reviews[chosen_reviews[vel_col] > 0]
            chosen_reviews = chosen_reviews[chosen_reviews[vol_col] > 0]
            chosen_reviews = chosen_reviews.sample(n=min(1000, len(chosen_reviews)))

            for rating in range(1, 6):
                axes[i, 0].scatter(
                    chosen_reviews[vel_col][chosen_reviews['review_rating'] == rating],
                    chosen_reviews[vol_col][chosen_reviews['review_rating'] == rating],
                    alpha=0.5,
                    label=f'Rating {rating}'
                )

            axes[i, 0].set_title(f'CL: {chunk_length}, SZ={chunk_size}')
            axes[i, 0].set_xlabel('Trace Velocity')
            axes[i, 0].set_ylabel('Trace Volume')
            axes[i, 0].legend()

            axes[i, 1].scatter(chosen_reviews[vel_col][chosen_reviews['is_from_cracow']],
                               chosen_reviews[vol_col][chosen_reviews['is_from_cracow']],
                               alpha=0.5, label='From Cracow')
            axes[i, 1].scatter(chosen_reviews[vel_col][~chosen_reviews['is_from_cracow']],
                               chosen_reviews[vol_col][~chosen_reviews['is_from_cracow']],
                               alpha=0.5, label='From Warsaw')
            axes[i, 1].set_title(f'CL: {chunk_length}, SZ={chunk_size}')
            axes[i, 1].set_xlabel('Trace Velocity')
            axes[i, 1].set_ylabel('Trace Volume')
            axes[i, 1].legend()

            axes[i, 2].scatter(chosen_reviews[vel_col][chosen_reviews['is_translated']],
                               chosen_reviews[vol_col][chosen_reviews['is_translated']],
                               alpha=0.5, label='Translated')
            axes[i, 2].scatter(chosen_reviews[vel_col][~chosen_reviews['is_translated']],
                               chosen_reviews[vol_col][~chosen_reviews['is_translated']],
                               alpha=0.5, label='Not Translated')
            axes[i, 2].set_title(f'CL: {chunk_length}, SZ={chunk_size}')
            axes[i, 2].set_xlabel('Trace Velocity')
            axes[i, 2].set_ylabel('Trace Volume')
            axes[i, 2].legend()

        figures['clustering_velocity_volume'] = fig

        return figures

    def _get_vol_vel_chunk_infos(self) -> Set[Tuple[int, int]]:

        chunk_infos: Set[Tuple[int, int]] = set()

        for col in self._numerical_stats.columns:
            vol_vel_match = re.match(r'trace_(velocity|volume)_cl_(\d+)_sz_(\d+)', col)
            if vol_vel_match:
                chunk_length = int(vol_vel_match.group(2))
                chunk_size = int(vol_vel_match.group(3))
                chunk_infos.add((chunk_length, chunk_size))

        return chunk_infos

    def _get_bbow_nzero_df(self) -> pd.DataFrame:

        def get_bbow_nzero(bbow_path: str) -> int:
            bbow = torch.load(bbow_path)
            return bbow.indices().size(1)

        sizes_df = self._paths_df.copy()

        sizes_df['top'] = sizes_df['bow_representations_top'].apply(get_bbow_nzero)
        sizes_df['bottom'] = sizes_df['bow_representations_bottom'].apply(get_bbow_nzero)
        sizes_df['full'] = sizes_df['bow_representations_full'].apply(get_bbow_nzero)

        return sizes_df[['top', 'bottom', 'full']]
