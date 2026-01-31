"""Module that handles access to directories and files related to processed datasets."""


import pathlib
import json
from typing import Iterator

import pydantic

from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.utils import misc as misc_utils


class ProcessedReview(pydantic.BaseModel):
    """Represents a processed review with its data features"""
    restaurant_info: raw_ds.Restaurant
    raw_review: raw_ds.Review
    normalized_text_pth: pathlib.Path
    word_count_vector_pth: pathlib.Path
    pos_count_vector_pth: pathlib.Path
    bert_embeddings_pth: pathlib.Path
    trace_features_pth: pathlib.Path

    def load_normalized_text(self) -> str:
        """Loads the normalized text of the review from file."""

        with self.normalized_text_pth.open('r', encoding='utf-8') as f:
            return f.read()


class ProcessedDsPathHandler:
    """Manages access to processed dataset directories and files."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    def get_path_to_restaurant(self, restaurant: raw_ds.Restaurant) -> pathlib.Path:
        """Returns the path to the directory for a specific restaurant."""
        return (
            self._base_path
            .joinpath('restaurants')
            .joinpath(misc_utils.hash_restaurant_href(restaurant.href))
        )

    def create_new_review(self,
                          restaurant: raw_ds.Restaurant,
                          raw_review: raw_ds.Review) -> ProcessedReview:
        """Creates a new ProcessedReview instance with file paths set."""

        restaurant_path = self.get_path_to_restaurant(restaurant)

        if not restaurant_path.exists():
            restaurant_path.mkdir(parents=True, exist_ok=True)

            with restaurant_path.joinpath('info.json').open('w', encoding='utf-8') as f:
                json.dump(restaurant.model_dump(), f, ensure_ascii=False, indent=4)

        review_path = (restaurant_path
                       .joinpath('reviews')
                       .joinpath(raw_ds.hash_review(raw_review)))

        review_path.mkdir(parents=True, exist_ok=True)

        with review_path.joinpath('raw_review.json').open('w', encoding='utf-8') as f:
            json.dump(raw_review.model_dump(), f, ensure_ascii=False, indent=4)

        return self._review_from_path(
            review_path=review_path,
            restaurant_info=restaurant,
            raw_review=raw_review
        )

    def iter_restaurants(self) -> Iterator[raw_ds.Restaurant]:
        """Loads all restaurants in the processed dataset."""

        for restaurant_dir in self._base_path.joinpath('restaurants').iterdir():

            with restaurant_dir.joinpath('info.json').open('r', encoding='utf-8') as f:
                restaurant_info = raw_ds.Restaurant.model_validate(json.load(f))

            yield restaurant_info

    def iter_reviews_for(self, restaurant: raw_ds.Restaurant) -> Iterator[ProcessedReview]:
        """Loads all processed reviews for a specific restaurant."""

        restaurant_dir = self.get_path_to_restaurant(restaurant)
        reviews_path = restaurant_dir.joinpath('reviews')

        for review_dir in reviews_path.iterdir():

            with review_dir.joinpath('raw_review.json').open('r', encoding='utf-8') as f:
                review = raw_ds.Review.model_validate(json.load(f))

            yield self._review_from_path(review_path=review_dir,
                                         restaurant_info=restaurant,
                                         raw_review=review)

    def _review_from_path(self,
                          review_path: pathlib.Path,
                          restaurant_info: raw_ds.Restaurant,
                          raw_review: raw_ds.Review) -> ProcessedReview:
        """Loads a ProcessedReview from its directory path."""

        return ProcessedReview(
            restaurant_info=restaurant_info,
            raw_review=raw_review,
            normalized_text_pth=review_path.joinpath('normalized_text.txt'),
            word_count_vector_pth=review_path.joinpath('word_count_vector.pt'),
            pos_count_vector_pth=review_path.joinpath('pos_count_vector.pt'),
            bert_embeddings_pth=review_path.joinpath('bert_embeddings.pt'),
            trace_features_pth=review_path.joinpath('trace_features.yaml')
        )
