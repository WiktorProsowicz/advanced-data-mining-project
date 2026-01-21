"""Tools for performing exploratory data analysis (EDA) on raw datasets."""

import pathlib
import json

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.data.eda import utils as eda_utils


class RawEDA:
    """Extracts statistics from raw datasets."""

    def __init__(self, raw_ds_path: str) -> None:

        self._raw_ds = raw_ds.RawDSLoader(raw_ds_path).load_dataset()

    def save_authors_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics about review authors in a given directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        def _df_generator():  # type: ignore
            for _, reviews in self._raw_ds.items():
                for review in reviews:

                    yield (review.author.n_reviews, review.rating)

        authors_df = pd.DataFrame(_df_generator(),  # type: ignore
                                  columns=['n_reviews', 'rating'])

        authors_df.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        ).to_markdown(f'{output_dir}/authors_stats.md')

        n_reviews_outliers = eda_utils.is_outlier(authors_df['n_reviews'])

        self._save_n_reviews_distribution(
            authors_df['n_reviews'],
            output_dir / 'n_reviews_distribution.svg')
        self._save_n_reviews_distribution(
            authors_df['n_reviews'][~n_reviews_outliers],
            output_dir / 'n_reviews_distribution_no_outliers.svg'
        )
        self._save_n_reviews_distribution(
            authors_df['n_reviews'][n_reviews_outliers],
            output_dir / 'n_reviews_distribution_only_outliers.svg'
        )

        self._save_rating_by_n_reviews_distribution(
            authors_df,
            output_dir / 'rating_by_n_reviews_distribution.svg'
        )
        self._save_rating_by_n_reviews_distribution(
            authors_df[~n_reviews_outliers],
            output_dir / 'rating_by_n_reviews_distribution_no_outliers.svg'
        )
        self._save_rating_by_n_reviews_distribution(
            authors_df[n_reviews_outliers],
            output_dir / 'rating_by_n_reviews_distribution_only_outliers.svg'
        )

    def save_locations_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics about review locations in a given directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        def _df_generator():  # type: ignore
            for location, reviews in self._raw_ds.items():
                for review in reviews:

                    yield (location.primary_location,
                           location.primary_location + location.secondary_location,
                           location.href,
                           review.rating,
                           review.author.n_reviews)

        locations_df = pd.DataFrame(_df_generator(),  # type: ignore
                                    columns=['primary_location', 'secondary_location',
                                             'restaurant_name', 'rating', 'n_reviews'])

        primary_stats = {
            'n_locations': locations_df['primary_location'].nunique(),
            'n_secondary_locations': locations_df['secondary_location'].nunique(),
            'n_restaurants': locations_df['restaurant_name'].nunique(),
            'n_reviews': len(locations_df),

            'n_sec_locs_by_prim_loc': locations_df.groupby(
                'primary_location')['secondary_location'].nunique().describe().to_dict(),

            'n_reviews_by_prim_loc': locations_df.groupby(
                'primary_location').size().describe().to_dict(),
            'n_reviews_by_sec_loc': locations_df.groupby(
                'secondary_location').size().describe().to_dict(),
            'n_reviews_by_restaurant': locations_df.groupby(
                'restaurant_name').size().describe().to_dict(),

            'n_restaurants_by_prim_loc': locations_df.groupby(
                'primary_location')['restaurant_name'].nunique().describe().to_dict(),
            'n_restaurants_by_sec_loc': locations_df.groupby(
                'secondary_location')['restaurant_name'].nunique().describe().to_dict(),
        }

        with output_dir.joinpath('locations_stats.json').open('w') as f:
            json.dump(primary_stats, f)

        self._save_distributions_by_location(locations_df, output_dir)

    def _save_n_reviews_distribution(self,
                                     n_reviews_series: pd.Series,
                                     output_path: pathlib.Path) -> None:
        """Saves distribution plot of number of reviews written by authors."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.violinplot(
            n_reviews_series,
            orient='v',
            inner='quart',
            inner_kws={'linewidth': 2},
            legend=False,
            color='steelblue',
            ax=ax
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.xaxis.set_ticks([])
        ax.set_title('Number of all Google Maps reviews written by authors')
        ax.set_ylabel('Number of reviews')

        fig.savefig(output_path)

    def _save_rating_by_n_reviews_distribution(self,
                                               authors_df: pd.DataFrame,
                                               output_path: pathlib.Path) -> None:
        """Saves distribution plot of rating by number of reviews written by the authors."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.violinplot(
            authors_df,
            x='rating',
            y='n_reviews',
            hue='rating',
            orient='v',
            inner='quart',
            inner_kws={'linewidth': 2},
            palette='coolwarm',
            legend=False,
            ax=ax
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.set_title('Distribution of number of authors\' reviews by review rating')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Number of authors\' reviews')

        fig.savefig(output_path)

    def _save_distributions_by_location(self,
                                        locations_df: pd.DataFrame,
                                        output_dir: pathlib.Path) -> None:
        """Saves various distribution plots grouped by primary location."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.countplot(locations_df,
                      x='primary_location',
                      color='steelblue',
                      legend=False,
                      ax=ax)

        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.set_title('Number of reviews by location')
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of reviews')

        fig.savefig(output_dir / 'n_reviews_by_location.svg')
