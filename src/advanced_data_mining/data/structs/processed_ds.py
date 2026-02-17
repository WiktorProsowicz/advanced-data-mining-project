"""Module that handles access to directories and files related to processed datasets."""


import pathlib
import json
from typing import Iterator

import pydantic
import torch

from advanced_data_mining.data.structs import raw_ds
from advanced_data_mining.utils import misc as misc_utils


class ProcessingMetadataPathHandler:
    """Manages access to processing metadata files."""

    def __init__(self, base_path: pathlib.Path) -> None:
        self._base_path = base_path

    @property
    def numerical_features_extractor_path(self) -> pathlib.Path:
        """Returns the path to the numerical features configuration file."""
        return self._base_path.joinpath('numerical_features_extractor/')

    @property
    def numerical_features_cfg_path(self) -> pathlib.Path:
        """Returns the path to the numerical features configuration file."""
        return self._base_path.joinpath('numerical_features_extractor/config.json')

    @property
    def bert_embeddings_cfg_path(self) -> pathlib.Path:
        """Returns the path to the BERT embeddings configuration file."""
        return self._base_path.joinpath('bert_embeddings_generator_cfg.json')

    @property
    def scaling_metadata_path(self) -> pathlib.Path:
        """Returns the directory containing scaling metadata."""
        return self._base_path.joinpath('scaling_metadata/')

    @property
    def count_vectorizer_path(self) -> pathlib.Path:
        """Returns the path to the count vectorizer metadata."""
        return self._base_path.joinpath('count_vectorizer/')

    def get_supported_cat_opt_names(self) -> list[str]:
        """Returns the list of supported categorical option names."""

        with self.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        return [
            option_setup['name'] for option_setup in cfg['categorized_options_used']
        ]

    def get_cat_opt_label_mapping(self, feature_name: str) -> dict[int, str]:
        """Returns the mapping from encoded categorical option values to labels."""

        with self.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        for option_setup in cfg['categorized_options_used']:
            if option_setup['name'] == feature_name:
                return {**dict(enumerate(option_setup['supported_values'], 1)),
                        0: 'UNKNOWN',
                        len(option_setup['supported_values']) + 1: 'OTHER'}

        return {}

    def get_n_author_reviews_label_mapping(self) -> dict[int, str]:
        """Returns the mapping for number of author reviews index to labels."""

        with self.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        return {
            **{index: quant['cat_name']
               for index, quant in enumerate(cfg['n_author_reviews_quantization'], 1)},
            len(cfg['n_author_reviews_quantization']) + 1: 'OTHER',
            0: 'UNKNOWN'
        }

    def get_chunk_and_step_sizes(self) -> list[tuple[int, int]]:
        """Returns the list of chunk sizes and step sizes for trace features extraction."""

        with self.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        return [tuple(chunk_setup) for chunk_setup in cfg['chunk_and_step_sizes']]

    def get_trace_scaling_params(self) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
        """Returns the scaling parameters for trace features."""

        scaling_params: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

        for chunk_size, step_size in self.get_chunk_and_step_sizes():

            pth = (self.scaling_metadata_path /
                   f'trace_features_mean_std_chunk_{chunk_size}_step_{step_size}.pt')

            scaling_params[(chunk_size, step_size)] = tuple(t.to(torch.float32)
                                                            for t in torch.load(pth))

        return scaling_params


class ProcessedReview(pydantic.BaseModel):
    """Represents a processed review with its data features"""
    restaurant_info: raw_ds.Restaurant
    raw_review: raw_ds.Review
    normalized_text_pth: pathlib.Path
    word_count_vector_pth: pathlib.Path
    pos_count_vector_pth: pathlib.Path
    bert_embeddings_pth: pathlib.Path
    trace_features_pth: pathlib.Path
    num_features_pth: pathlib.Path

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

    def iter_all_reviews(self) -> Iterator[ProcessedReview]:
        """Loads all processed reviews in the dataset."""

        for restaurant in self.iter_restaurants():
            yield from self.iter_reviews_for(restaurant)

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
            trace_features_pth=review_path.joinpath('trace_features.json'),
            num_features_pth=review_path.joinpath('numerical_features.json')
        )
