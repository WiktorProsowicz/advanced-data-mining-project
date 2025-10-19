"""Contains utilities for exploratory data analysis (EDA)."""

from typing import Dict, Any
import os

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore


class EDAFeatureExtractor:
    """Extracts features from the dataset for EDA purposes."""

    def __init__(self, processed_ds_path: str):

        self._processed_ds_path = processed_ds_path

    def extract_basic_stats(self) -> Dict[str, Any]:
        """Extracts basic statistics from the dataset."""

        stats: Dict[str, Any] = {}

        ds = pd.read_pickle(os.path.join(self._processed_ds_path, 'preprocessed_dataset.pkl'))
        numerical_stats = pd.read_pickle(os.path.join(self._processed_ds_path,
                                                      'numerical_features.pkl'))

        stats['total_restaurants'] = int(ds['restaurant_href'].nunique())
        stats['total_reviews'] = int(len(ds))
        stats['avg_review'] = float(stats['total_reviews'] / stats['total_restaurants'])

        restaurant_ratings = ds.groupby('restaurant_href')['review_rating'].mean()

        stats['avg_restaurant_rating'] = float(restaurant_ratings.mean())
        stats['min_restaurant_rating'] = float(restaurant_ratings.min())
        stats['max_restaurant_rating'] = float(restaurant_ratings.max())

        stats['avg_words_in_review'] = float(numerical_stats['num_words'].mean())
        stats['min_words_in_review'] = float(numerical_stats['num_words'].min())
        stats['max_words_in_review'] = float(numerical_stats['num_words'].max())

        stats['avg_sentences_in_review'] = float(numerical_stats['num_sentences'].mean())
        stats['min_sentences_in_review'] = float(numerical_stats['num_sentences'].min())
        stats['max_sentences_in_review'] = float(numerical_stats['num_sentences'].max())

        stats['n_reviews_from_cracow'] = int(len(ds[ds['is_from_cracow']]))

        for col in [c for c in numerical_stats.columns if c.startswith('trace_velocity_')]:
            stats[f'avg_{col}'] = float(numerical_stats[col].mean())
            stats[f'min_{col}'] = float(numerical_stats[col].min())
            stats[f'max_{col}'] = float(numerical_stats[col].max())

        for col in [c for c in numerical_stats.columns if c.startswith('trace_volume_')]:
            stats[f'avg_{col}'] = float(numerical_stats[col].mean())
            stats[f'min_{col}'] = float(numerical_stats[col].min())
            stats[f'max_{col}'] = float(numerical_stats[col].max())

        return stats

    def get_figures(self) -> Dict[str, plt.Figure]:
        """Generates figures for EDA."""

        figures: Dict[str, plt.Figure] = {}

        ds = pd.read_pickle(os.path.join(self._processed_ds_path, 'preprocessed_dataset.pkl'))
        numerical_stats = pd.read_pickle(os.path.join(self._processed_ds_path,
                                                      'numerical_features.pkl'))

        fig, ax = plt.subplots()

        ax.hist(ds['review_rating'], bins=5, range=(1, 6), align='left', rwidth=0.8)
        ax.set_title('Distribution of Review Ratings')
        ax.set_xlabel('Review Rating')
        ax.set_ylabel('Number of Reviews')

        figures['review_rating_distribution'] = fig

        fig, ax = plt.subplots()

        ax.hist(numerical_stats['num_words'], bins=100, color='orange')
        ax.set_title('Distribution of Number of Words in Reviews')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Number of Reviews')

        figures['num_words_distribution'] = fig

        fig, ax = plt.subplots()

        ax.hist(numerical_stats['num_sentences'], bins=50, color='green')
        ax.set_title('Distribution of Number of Sentences in Reviews')
        ax.set_xlabel('Number of Sentences')
        ax.set_ylabel('Number of Reviews')

        figures['num_sentences_distribution'] = fig

        return figures
