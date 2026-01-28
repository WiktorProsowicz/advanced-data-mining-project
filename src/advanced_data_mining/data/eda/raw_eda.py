"""Tools for performing exploratory data analysis (EDA) on raw datasets."""

import pathlib
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.data.eda import utils as eda_utils
from advanced_data_mining.data.processing import num_features


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
                           review.author.n_reviews,
                           review.original is not None)
        locations_df = pd.DataFrame(_df_generator(),  # type: ignore
                                    columns=['primary_location', 'secondary_location',
                                             'restaurant_name', 'rating', 'n_reviews',
                                             'is_translated'])

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

        self._save_proportion_translated_reviews_by_location_size(locations_df, output_dir)

    def save_review_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics about reviews in a given directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        def _df_generator():  # type: ignore
            for _, reviews in self._raw_ds.items():
                for review in reviews:

                    yield (int(review.rating),
                           len(review.text),
                           review.original is not None)

        reviews_df = pd.DataFrame(_df_generator(),  # type: ignore
                                  columns=['rating', 'text_length', 'is_translated'])

        primary_stats = {
            'n_reviews': len(reviews_df),
            'rating_stats': reviews_df['rating'].describe().to_dict(),
            'text_length_stats': reviews_df['text_length'].describe().to_dict(),
            'n_translated_reviews': int(reviews_df['is_translated'].sum()),
        }

        with output_dir.joinpath('review_stats.json').open('w') as f:
            json.dump(primary_stats, f, indent=4)

        self._save_review_rating_distribution(reviews_df, output_dir)

        self._save_review_length_distribution(
            reviews_df[~eda_utils.is_outlier(reviews_df['text_length'])],
            output_dir / 'review_length_distribution_no_outliers.svg')

        self._save_categorized_options_stats(output_dir / 'categorized_options_stats/')

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
            color=eda_utils.MIDDLE_COLOR_STD,
            ax=ax
        )

        ax.set_axisbelow(True)
        ax.grid(axis='x', which='minor')
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
            palette=eda_utils.get_gradient_palette(5),
            legend=False,
            ax=ax
        )

        ax.set_axisbelow(True)
        ax.grid(axis='x', which='minor')
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
                      color=eda_utils.MIDDLE_COLOR_STD,
                      stat='percent',
                      legend=False,
                      width=1,
                      ax=ax)

        ax.set_axisbelow(True)
        ax.grid(axis='y', which='minor')
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
                color=eda_utils.get_gradient_cmap()((rating - 1) / 4)
            )

        ax.set_axisbelow(True)
        ax.grid(axis='y', which='minor')
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
            palette=eda_utils.get_coolwarm_cmap(),
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
            palette=eda_utils.get_coolwarm_cmap(),
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

    def _save_proportion_translated_reviews_by_location_size(self,
                                                             locations_df: pd.DataFrame,
                                                             output_dir: pathlib.Path) -> None:
        """Saves plot of proportion of translated reviews by location size."""

        stats_df = (
            locations_df
            .groupby('primary_location', as_index=False)
            .agg(n_reviews=('is_translated', 'size'),
                 n_translated_reviews=('is_translated', 'sum'))
        )
        stats_df['Proportion of translated reviews'] = (
            stats_df['n_translated_reviews'] / stats_df['n_reviews']
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
            y='Proportion of translated reviews',
            hue='Proportion of translated reviews',
            palette=eda_utils.get_gradient_cmap(),
            ax=ax
        )

        ax.set_title('Proportion of translated reviews by location size.')
        ax.set_ylabel('Proportion of translated reviews')
        ax.set_xlabel('Number of restaurants in location')
        ax.xaxis.grid(True, 'minor', linewidth=0.25)
        ax.set_xscale('log')
        ax.set_axisbelow(True)
        fig.subplots_adjust(top=.95)

        fig.savefig(output_dir / 'proportion_translated_reviews_by_location_size.svg')

    def _save_review_rating_distribution(self,
                                         reviews_df: pd.DataFrame,
                                         output_dir: pathlib.Path) -> None:
        """Saves distribution plot of review ratings."""

        graph = sns.catplot(
            data=reviews_df,
            x='rating',
            kind='count',
            hue='is_translated',
            palette=eda_utils.get_gradient_palette(2),
            height=6,
            aspect=1.5
        )

        graph.set_axis_labels('Review rating', 'Number of reviews')
        graph.ax.set_title('Distribution of review ratings')
        graph.ax.set_axisbelow(True)
        graph.ax.grid(axis='y', which='minor')
        graph.legend.set_title('Is translated')
        graph.figure.subplots_adjust(top=.95)

        graph.figure.savefig(output_dir / 'review_rating_distribution.svg')

    def _save_review_length_distribution(self,
                                         reviews_df: pd.DataFrame,
                                         output_path: pathlib.Path) -> None:
        """Saves distribution plot of review text lengths."""

        fig, ax = plt.subplots(figsize=(10, 6))

        reviews_df = reviews_df.rename(columns={'text_length': 'Review text length',
                                                'rating': 'Review rating',
                                                'is_translated': 'Is translated'})

        sns.violinplot(
            data=reviews_df,
            x='Review rating',
            y='Review text length',
            hue='Is translated',
            split=True,
            inner="quart",
            palette=eda_utils.get_gradient_palette(2),
            fill=False,
            ax=ax
        )

        ax.set_title('Review text length vs. review rating')
        ax.set_xlabel('Review rating')
        ax.set_ylabel('Review text length (characters)')
        ax.xaxis.grid(True, 'minor', linewidth=0.25)
        ax.set_axisbelow(True)
        fig.subplots_adjust(top=.95)

        fig.savefig(output_path)

    def _save_categorized_options_stats(self, output_dir: pathlib.Path) -> None:
        """Saves statistics about categorized opinions in reviews."""

        def _raw_df_generator():  # type: ignore
            for _, reviews in self._raw_ds.items():
                for review in reviews:

                    if review.categorized_opinions is not None:
                        yield review.categorized_opinions
                    else:
                        yield {}

        self._save_categorized_options_stats_for_df(
            pd.DataFrame(_raw_df_generator()),  # type: ignore
            output_dir / 'raw/'
        )

        def _sanitized_df_generator():  # type: ignore
            for _, reviews in self._raw_ds.items():
                for review in reviews:

                    if review.categorized_opinions is not None:
                        yield num_features.sanitize_categorized_options(review.categorized_opinions)
                    else:
                        yield {}

        self._save_categorized_options_stats_for_df(
            pd.DataFrame(_sanitized_df_generator()),  # type: ignore
            output_dir / 'sanitized/'
        )

    def _save_categorized_options_stats_for_df(self,
                                               cat_opts_df: pd.DataFrame,
                                               output_dir: pathlib.Path) -> None:
        """Saves statistics about categorized opinions in reviews from a given DataFrame."""

        output_dir.mkdir(parents=True, exist_ok=True)

        basic_stats = {
            **cat_opts_df.describe().to_dict(),
            'n_reviews_with_no_categorized_options': int(cat_opts_df.isna().all(axis=1).sum())
        }

        with output_dir.joinpath('categorized_options_stats.json').open('w') as f:
            json.dump(basic_stats, f, indent=4, ensure_ascii=False)

        for option_name in cat_opts_df.columns:
            self._save_distribution_of_unique_values_of_cat_option(
                cat_opts_df,
                option_name,
                output_dir
            )

    def _save_distribution_of_unique_values_of_cat_option(self,
                                                          cat_opts_df: pd.DataFrame,
                                                          option_name: str,
                                                          output_dir: pathlib.Path) -> None:
        """Saves distribution of unique values of a categorized option."""

        fig, ax = plt.subplots(figsize=(10, 10))

        dist_df = cat_opts_df[option_name].value_counts().reset_index()
        dist_df.sort_values(by='count', ascending=False, inplace=True)

        original_count = len(dist_df)
        dist_df = dist_df.iloc[:min(50, original_count)]

        sns.barplot(
            x=dist_df['count'],
            y=dist_df[option_name],
            color=eda_utils.MIDDLE_COLOR_STD,
            ax=ax,
            orient='y',
            width=1
        )

        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.set_title(
            f'Distribution of {len(dist_df)}/{original_count} most common\n'
            f'values of categorized option "{option_name}"')
        ax.set_xlabel('Number of occurrences')
        ax.set_ylabel('Categorized option value')

        fig.subplots_adjust(left=.3)
        fig.savefig(output_dir / f'{option_name.replace(" ", "_")}_distribution.svg')
