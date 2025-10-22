"""Contains utilities for exploratory data analysis (EDA)."""

from typing import Dict, Any, List, Set, Tuple
import os
import re

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore


class EDAFeatureExtractor:
    """Extracts features from the dataset for EDA purposes."""

    def __init__(self, processed_ds_path: str):

        self._processed_ds_path = processed_ds_path

        self._ds = pd.read_pickle(os.path.join(self._processed_ds_path, 'preprocessed_dataset.pkl'))
        self._numerical_stats = pd.read_pickle(os.path.join(self._processed_ds_path,
                                                            'numerical_features.pkl'))

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

        return stats

    def get_figures(self) -> Dict[str, plt.Figure]:
        """Generates figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        figures.update(self._get_distribution_figures())

        figures.update(self._get_clustering_figures())

        return figures

    def _get_distribution_figures(self) -> Dict[str, plt.Figure]:
        """Generates distribution figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        fig, ax = plt.subplots()

        ax.hist(self._ds['review_rating'], bins=5, range=(1, 6), align='left', rwidth=0.8)
        ax.set_title('Distribution of Review Ratings')
        ax.set_xlabel('Review Rating')
        ax.set_ylabel('Number of Reviews')

        figures['review_rating_distribution'] = fig

        fig, ax = plt.subplots()

        ax.hist(self._numerical_stats['num_words'], bins=100, color='orange')
        ax.set_title('Distribution of Number of Words in Reviews')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Number of Reviews')

        figures['num_words_distribution'] = fig

        fig, ax = plt.subplots()

        ax.hist(self._numerical_stats['num_sentences'], bins=50, color='green')
        ax.set_title('Distribution of Number of Sentences in Reviews')
        ax.set_xlabel('Number of Sentences')
        ax.set_ylabel('Number of Reviews')

        figures['num_sentences_distribution'] = fig

        return figures

    def _get_clustering_figures(self) -> Dict[str, plt.Figure]:
        """Generates clustering figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        for chunk_length, chunk_size in self._get_vol_vel_chunk_infos():

            fig, ax = plt.subplots()

            vel_col = f'trace_velocity_cl_{chunk_length}_sz_{chunk_size}'
            vol_col = f'trace_volume_cl_{chunk_length}_sz_{chunk_size}'

            ax.scatter(self._numerical_stats[vel_col],
                       self._numerical_stats[vol_col],
                       alpha=0.5,
                       c=self._numerical_stats['review_rating'])
            ax.set_title(f'Clustering of Velocity vs Volume (Chunk Length: {chunk_length})')
            ax.set_xlabel('Trace Velocity')
            ax.set_ylabel('Trace Volume')

            figures[f'clustering_vel_vs_vol_cl_{chunk_length}'] = fig

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
