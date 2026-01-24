"""Tools for performing exploratory data analysis (EDA) on raw datasets."""

import pathlib
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.data.eda import utils as eda_utils


class RawEDA:
    """Extracts statistics from raw datasets."""

    def __init__(self, raw_ds_path: str) -> None:

        self._raw_ds = raw_ds.RawDSLoader(raw_ds_path).load_dataset()
        sns.set_theme(style="darkgrid")

    def save_authors_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics about review authors in a given directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        def _df_generator():  # type: ignore
            for _, reviews in self._raw_ds.items():
                for review in reviews:

                    yield (review.author.n_reviews, int(review.rating))

        authors_df = pd.DataFrame(_df_generator(),  # type: ignore
                                  columns=['n_reviews', 'rating'])

        authors_df.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        ).to_markdown(f'{output_dir}/authors_stats.md')

        n_reviews_outliers = eda_utils.is_outlier(authors_df['n_reviews'])

        self._save_written_reviews_distribution(
            authors_df['n_reviews'],
            output_dir / 'n_reviews_distribution.svg')
        self._save_written_reviews_distribution(
            authors_df['n_reviews'][~n_reviews_outliers],
            output_dir / 'n_reviews_distribution_no_outliers.svg'
        )

        self._save_written_reviews_by_rating_distribution(
            authors_df,
            output_dir / 'n_reviews_by_rating_distribution.svg'
        )
        self._save_written_reviews_by_rating_distribution(
            authors_df[~n_reviews_outliers],
            output_dir / 'n_reviews_by_rating_distribution_no_outliers.svg'
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
                           int(review.rating),
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
            json.dump(primary_stats, f, indent=4)

        self._save_percent_reviews_by_location(locations_df, output_dir)

        self._save_rating_distribution_by_location(locations_df, output_dir)

        self._save_distributions_by_location_size(locations_df, output_dir)

        self._save_restaurant_rating_by_location_size_scatter(locations_df, output_dir)

    def _save_written_reviews_distribution(self,
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

    def _save_written_reviews_by_rating_distribution(self,
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

    def _save_percent_reviews_by_location(self,
                                          locations_df: pd.DataFrame,
                                          output_dir: pathlib.Path) -> None:
        """Saves distribution of all reviews by primary location."""

        fig, ax = plt.subplots(figsize=(6, 10))

        sns.countplot(locations_df,
                      order=locations_df['primary_location'].value_counts().index,
                      y='primary_location',
                      color='steelblue',
                      stat='percent',
                      legend=False,
                      width=1,
                      ax=ax)

        ax.set_axisbelow(True)
        ax.grid(axis='x')
        ax.set_title('Distribution of reviews by locations')
        ax.set_ylabel('Location')
        ax.set_xlabel('Percent of reviews')

        fig.tight_layout()
        fig.savefig(output_dir / 'percent_reviews_by_location.svg')

    def _save_rating_distribution_by_location(self, locations_df: pd.DataFrame,
                                              output_dir: pathlib.Path) -> None:
        """Saves deistributions of ratings grouped by primary location."""

        fig, ax = plt.subplots(figsize=(6, 10))

        ratings = [1, 2, 3, 4, 5]

        review_counts = pd.DataFrame(
            {loc: locations_df[locations_df['primary_location'] == loc]['rating'].value_counts()
                for loc in locations_df['primary_location'].unique()}
        ).T
        review_counts['primary_location'] = review_counts.index
        review_counts.reset_index(drop=True, inplace=True)
        review_counts[ratings] = review_counts[ratings].div(review_counts[ratings].sum(axis=1),
                                                            axis=0)

        for i, rating in enumerate(reversed(ratings)):

            review_counts[rating] = review_counts[ratings[:5 - i]].sum(axis=1)

            sns.barplot(
                data=review_counts,
                y='primary_location',
                x=rating,
                label=str(rating),
                ax=ax,
                width=1,
                color=plt.get_cmap('Blues')(.5 + (rating - 1) / 4 / 2)
            )

        ax.set_axisbelow(True)
        ax.grid(axis='x')
        ax.set_title('Distribution of reviews by locations')
        ax.set_ylabel('Location')
        ax.set_xlabel('Percent of reviews')
        ax.legend(title='Rating')
        ax.set_xlim(0, 1)

        fig.tight_layout()

        fig.savefig(output_dir / 'rating_distribution_by_location.svg')

    def _save_distributions_by_location_size(self,
                                             locations_df: pd.DataFrame,
                                             output_dir: pathlib.Path) -> None:
        """Saves distributions of restaurant ratings and written review counts by location size."""

        stats_df = (
            locations_df
            .groupby(['primary_location', 'restaurant_name'], as_index=False)
            .agg(restaurant_rating=('rating', 'mean'))
            .groupby('primary_location', as_index=True)
            .agg(mean_restaurant_rating=('restaurant_rating', 'mean'))
        ).rename(columns={'mean_restaurant_rating': 'Mean restaurant rating'})

        stats_df['Number of reviews in location'] = (
            locations_df
            .groupby('primary_location')
            .size()
            .reindex(stats_df.index)
        )
        stats_df['Avg number of written reviews'] = (
            locations_df
            .groupby('primary_location')['n_reviews']
            .mean()
            .reindex(stats_df.index)
        )
        stats_df['Number of restaurants in location'] = (
            locations_df
            .groupby('primary_location')['restaurant_name']
            .nunique()
            .reindex(stats_df.index)
        )

        graph = sns.relplot(
            data=stats_df,
            x='Number of reviews in location',
            y='Number of restaurants in location',
            size='Avg number of written reviews',
            hue='Mean restaurant rating',
            palette='coolwarm',
            alpha=0.7,
            height=6,
            aspect=1.5
        )

        graph.set(xscale='log', yscale='log')
        graph.ax.set_title(
            'Distribution of written reviews and mean restaurant ratings by location size.')
        graph.ax.yaxis.grid(True, 'minor', linewidth=0.25)
        graph.ax.xaxis.grid(True, 'minor', linewidth=0.25)
        graph.ax.set_axisbelow(True)
        graph.figure.subplots_adjust(top=.95)

        graph.figure.savefig(output_dir / 'distributions_by_location_size.svg')

    def _save_restaurant_rating_by_location_size_scatter(self,
                                                         locations_df: pd.DataFrame,
                                                         output_dir: pathlib.Path) -> None:
        """Saves scatter plot of restaurant ratings by location size."""

        stats_df = (
            locations_df
            .groupby(['primary_location', 'restaurant_name'], as_index=False)
            .agg(restaurant_rating=('rating', 'mean'))
            .rename(columns={'restaurant_rating': 'Restaurant rating'})
        )
        loc_counts = (
            locations_df
            .groupby('primary_location')['restaurant_name']
            .nunique()
        )
        stats_df['Number of restaurants in location'] = (
            stats_df['primary_location'].map(loc_counts)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.scatterplot(
            data=stats_df,
            x='Number of restaurants in location',
            y='Restaurant rating',
            hue='Restaurant rating',
            palette='coolwarm',
            ax=ax
        )

        ax.set_title('Restaurant rating by location size.')
        ax.set_ylabel('Restaurant rating')
        ax.set_xlabel('Number of restaurants in location')

        ax.xaxis.grid(True, 'minor', linewidth=0.25)
        ax.yaxis.grid(True, 'minor', linewidth=0.25)
        ax.set_xscale('log')
        ax.set_axisbelow(True)
        fig.subplots_adjust(top=.95)

        fig.savefig(output_dir / 'restaurant_rating_by_location_size.svg')
