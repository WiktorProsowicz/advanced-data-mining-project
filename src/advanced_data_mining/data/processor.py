"""Module that processes raw dataset into a structured processed dataset."""

import pathlib
import torch

from advanced_data_mining.data.processing.count_vectorizer import CountVectorizer
from advanced_data_mining.data.structs import raw_ds as raw_ds_structs
from advanced_data_mining.data.structs import processed_ds as processed_ds_structs
from advanced_data_mining.data.processing import text_processing


class DataProcessor:
    """Processes raw data into a structured format suitable for training/testing."""

    def __init__(self, count_vectorizer: CountVectorizer) -> None:

        self._count_vectorizer = count_vectorizer

    def fit_transform(self,
                      raw_dataset: raw_ds_structs.RawDataset,
                      output_dir: pathlib.Path) -> None:
        """Fits the processor on the raw dataset and transforms it."""

        self._normalize_and_save_review_drafts(
            raw_dataset,
            output_dir
        )

        path_handler = processed_ds_structs.ProcessedDsPathHandler(output_dir)

        for restaurant in path_handler.iter_restaurants():

            reviews = list(path_handler.iter_reviews_for(restaurant))
            self._count_vectorizer.fit([review.normalized_text for review in reviews])

        for restaurant in path_handler.iter_restaurants():

            reviews = list(path_handler.iter_reviews_for(restaurant))

            word_count_matrix = self._count_vectorizer.transform(
                [review.normalized_text for review in reviews]
            )

            for review, word_count_vector in zip(reviews, word_count_matrix):

                with review.word_count_vector_pth.open('wb') as f:
                    torch.save(torch.tensor(word_count_vector), f)

    def _normalize_and_save_review_drafts(self,
                                          raw_dataset: raw_ds_structs.RawDataset,
                                          processed_ds_path: pathlib.Path) -> None:
        """Normalizes and saves preprocessed review drafts to the specified directory."""

        path_handler = processed_ds_structs.ProcessedDsPathHandler(processed_ds_path)

        for restaurant, reviews in raw_dataset.items():

            for review in reviews:
                normalized_text = text_processing.normalize_text(review.text)

                path_handler.create_new_review(
                    restaurant=restaurant,
                    normalized_text=normalized_text,
                    is_translated=review.original is not None
                )
