"""Module that processes raw dataset into a structured processed dataset."""

import json
import logging
import pathlib
import yaml
import tqdm

import torch
from sklearn.preprocessing import StandardScaler

from advanced_data_mining.data.processing import count_vectorizer
from advanced_data_mining.data.processing import embeddings
from advanced_data_mining.data.processing import num_features
from advanced_data_mining.data.processing import utils as processing_utils
from advanced_data_mining.data.structs import raw_ds as raw_ds_structs
from advanced_data_mining.data.structs import processed_ds as processed_ds_structs


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class DataProcessor:
    """Processes raw data into a structured format suitable for training/testing."""

    def __init__(self,
                 vectorizer: count_vectorizer.CountVectorizer,
                 embeddings_generator: embeddings.EmbeddingGenerator,
                 num_features_extractor: num_features.NumericalFeaturesExtractor
                 ) -> None:

        self._count_vectorizer = vectorizer
        self._embeddings_generator = embeddings_generator
        self._num_features_extractor = num_features_extractor

        self._count_vectors_scaler = StandardScaler()
        self._pos_vectors_scaler = StandardScaler()

    def fit_transform(self,
                      raw_dataset: raw_ds_structs.RawDataset,
                      output_dir: pathlib.Path) -> None:
        """Fits the processor on the raw dataset and transforms it."""

        _logger().info('Fitting and transforming the dataset...')

        self._normalize_and_save_review_drafts(
            raw_dataset,
            output_dir
        )

        path_handler = processed_ds_structs.ProcessedDsPathHandler(output_dir)

        for restaurant in path_handler.iter_restaurants():

            self._count_vectorizer.fit([review.load_normalized_text()
                                        for review
                                        in path_handler.iter_reviews_for(restaurant)])

        self._generate_count_features(output_dir)

        self._generate_bert_features(output_dir)

        self._generate_numeric_features(output_dir)

    def transform(self,
                  raw_dataset: raw_ds_structs.RawDataset,
                  output_dir: pathlib.Path) -> None:
        """Transforms the raw dataset using the already fitted processor."""

        _logger().info('Transforming the dataset...')

        self._normalize_and_save_review_drafts(
            raw_dataset,
            output_dir
        )

        self._generate_count_features(output_dir)

        self._generate_bert_features(output_dir)

        self._generate_numeric_features(output_dir)

    def save_processing_metadata(self, output_dir: pathlib.Path) -> None:
        """Saves the processing metadata to the specified directory."""

        metadata_path_handler = processed_ds_structs.ProcessingMetadataPathHandler(output_dir)

        self._count_vectorizer.serialize(metadata_path_handler.count_vectorizer_path)

        with (metadata_path_handler.numerical_features_cfg_path
              .open('w', encoding='utf-8')) as f:
            f.write(self._num_features_extractor.cfg.model_dump_json())

        with (metadata_path_handler.bert_embeddings_cfg_path
              .open('w', encoding='utf-8')) as f:
            f.write(self._embeddings_generator.cfg.model_dump_json())

        metadata_path_handler.scaling_metadata_path.mkdir(parents=True, exist_ok=True)

        with (metadata_path_handler.scaling_metadata_path
              .joinpath('count_vectors_mean_std.pt')
              .open('wb')) as f:
            torch.save((torch.tensor(self._count_vectors_scaler.mean_),
                        torch.tensor(self._count_vectors_scaler.scale_)), f)

        with (metadata_path_handler.scaling_metadata_path
              .joinpath('pos_vectors_mean_std.pt')
              .open('wb')) as f:
            torch.save((torch.tensor(self._pos_vectors_scaler.mean_),
                        torch.tensor(self._pos_vectors_scaler.scale_)), f)

    def _generate_count_features(self, processed_ds_path: pathlib.Path) -> None:
        """Generates features based on count vectorization."""

        path_handler = processed_ds_structs.ProcessedDsPathHandler(processed_ds_path)

        for restaurant in tqdm.tqdm(path_handler.iter_restaurants(),
                                    desc='Generating count-based features',
                                    unit='restaurant'):

            reviews = list(path_handler.iter_reviews_for(restaurant))

            word_count_matrix = self._count_vectorizer.generate_word_count_vectors(
                [review.load_normalized_text() for review in reviews]
            )

            for review, word_count_vector in zip(reviews, word_count_matrix):

                self._count_vectors_scaler.partial_fit([word_count_vector])

                with review.word_count_vector_pth.open('wb') as f:
                    torch.save(torch.tensor(word_count_vector), f)

            pos_count_matrix = self._count_vectorizer.generate_pos_count_vectors(
                [review.load_normalized_text() for review in reviews]
            )

            for review, pos_count_vector in zip(reviews, pos_count_matrix):

                self._pos_vectors_scaler.partial_fit([pos_count_vector])

                with review.pos_count_vector_pth.open('wb') as f:
                    torch.save(torch.tensor(pos_count_vector), f)

    def _generate_bert_features(self, processed_ds_path: pathlib.Path) -> None:
        """Generates BERT embeddings and features based on them."""

        path_handler = processed_ds_structs.ProcessedDsPathHandler(processed_ds_path)

        for restaurant in tqdm.tqdm(path_handler.iter_restaurants(),
                                    desc='Generating BERT-based features',
                                    unit='restaurant'):

            for review in path_handler.iter_reviews_for(restaurant):

                word_embeddings, sentence_embeddings = (
                    self._embeddings_generator.get_bert_embeddings(review.load_normalized_text())
                )

                with review.bert_embeddings_pth.open('wb') as f:
                    torch.save(torch.stack(sentence_embeddings), f)

                trace_features = self._num_features_extractor.generate_trace_features(
                    word_embeddings=torch.cat(word_embeddings, dim=0)
                )

                with review.trace_features_pth.open('w', encoding='utf-8') as f:
                    yaml.dump(trace_features, f)

    def _generate_numeric_features(self, processed_ds_path: pathlib.Path) -> None:
        """Generates numerical features for the processed dataset."""

        path_handler = processed_ds_structs.ProcessedDsPathHandler(processed_ds_path)

        for restaurant in tqdm.tqdm(path_handler.iter_restaurants(),
                                    desc='Generating numerical features',
                                    unit='restaurant'):

            for review in path_handler.iter_reviews_for(restaurant):

                if review.raw_review.categorized_opinions is None:
                    cat_options = {}
                else:
                    cat_options = review.raw_review.categorized_opinions

                encoded_cat_options = (self._num_features_extractor
                                       .generate_cat_options_onehot_indices(cat_options))

                n_author_reviews_index = (self._num_features_extractor
                                          .generate_n_author_reviews_onehot_index(
                                              review.raw_review.author.n_reviews))

                with review.num_features_pth.open('w', encoding='utf-8') as f:
                    json.dump({
                        'encoded_cat_options': encoded_cat_options,
                        'n_author_reviews_index': n_author_reviews_index,
                        'is_translated': review.raw_review.original is not None
                    }, f)

    def _normalize_and_save_review_drafts(self,
                                          raw_dataset: raw_ds_structs.RawDataset,
                                          processed_ds_path: pathlib.Path) -> None:
        """Normalizes and saves preprocessed review drafts to the specified directory."""

        path_handler = processed_ds_structs.ProcessedDsPathHandler(processed_ds_path)

        for restaurant, reviews in tqdm.tqdm(raw_dataset.items(),
                                             desc='Preparing review drafts',
                                             unit='restaurant'):

            for review in tqdm.tqdm(reviews,
                                    desc=f'Normalizing reviews for {restaurant.name}',
                                    unit='review',
                                    leave=False):
                normalized_text = processing_utils.normalize_text(review.text)

                review_draft = path_handler.create_new_review(
                    restaurant=restaurant,
                    raw_review=review
                )

                with review_draft.normalized_text_pth.open('w', encoding='utf-8') as f:
                    f.write(normalized_text)
