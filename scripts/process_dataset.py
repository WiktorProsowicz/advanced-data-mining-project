"""Transforms raw dataset into a processed format suitable for model training.

The script performs the following steps:
1. Normalization of review texts.
2. Vocabulary building from the normalized texts.
3. Generation of Bag-of-Words (BOW) representations for each review based on top/bottom N words
    from the vocabulary.
4. Generation of basic numerical features for each review.
5. Generation of features derived from advanced text representations (e.g., BERT embeddings).
"""

import logging
import os
import hashlib
import json
from typing import Dict, List

import hydra
import pandas as pd  # type: ignore
import omegaconf
import tqdm  # type: ignore
import torch

from advanced_data_mining.data import eda
from advanced_data_mining.data import text_processing
from advanced_data_mining.utils import logging_utils
from advanced_data_mining.data import raw_ds


def _logger():
    return logging.getLogger(__name__)


def _obtain_preprocessed_ds(raw_dataset: raw_ds.RawDataset,
                            text_processor: text_processing.TextPreprocessor,
                            output_path: str) -> pd.DataFrame:

    preprocessed_ds_path = os.path.join(output_path, "preprocessed_dataset.pkl")

    if os.path.exists(preprocessed_ds_path):
        return pd.read_pickle(preprocessed_ds_path)

    preprocessed_reviews = []

    for restaurant, reviews in tqdm.tqdm(raw_dataset.items(),
                                         desc="Processing restaurants",
                                         total=len(raw_dataset)):

        for review in tqdm.tqdm(reviews,
                                desc="Processing reviews",
                                total=len(reviews),
                                leave=False):

            preprocessed_text = text_processor.normalize_text(review.text)

            if not preprocessed_text:
                _logger().debug('Skipping empty review for restaurant %s', restaurant.href)
                continue

            preprocessed_reviews.append(
                (restaurant.href,
                 restaurant.name,
                 restaurant.basic_info,
                 restaurant.city == "Krakow",
                 preprocessed_text,
                 review.rating)
            )

    prep_ds = pd.DataFrame(preprocessed_reviews,
                           columns=["restaurant_href", "restaurant_name", "restaurant_basic_info",
                                    "is_from_cracow", "review_text", "review_rating"])

    prep_ds.to_pickle(preprocessed_ds_path)

    return prep_ds


def _prepare_bow_representations(vocabulary_path: str,
                                 dataset: pd.DataFrame,
                                 output_path: str,
                                 use_tfidf: bool):
    """Prepares and saves BOW representations for the dataset."""

    text_processor = text_processing.TextPreprocessor()
    text_processor.load_vocab_from_file(vocabulary_path)

    _logger().info('Preparing BOW representations...')

    os.makedirs(output_path, exist_ok=True)

    for idx, row in dataset.iterrows():
        if use_tfidf:
            bow_repr = text_processor.get_tfidf_representation(row['review_text'])
        else:
            bow_repr = text_processor.get_bow_representation(row['review_text'])

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()
        os.makedirs(os.path.join(output_path, str(hashed_dir)), exist_ok=True)

        torch.save(bow_repr,
                   os.path.join(output_path, str(hashed_dir), f'{idx}.pt'))


def _prepare_numerical_features(dataset: pd.DataFrame,
                                text_processor: text_processing.TextPreprocessor,
                                output_dir: str):
    """Prepares and saves numerical features for the dataset."""

    _logger().info('Preparing numerical features...')

    chunk_lengths = (5, 7, 9, 11, 13)

    velocity_series: Dict[int, List[float]] = {cl: [] for cl in chunk_lengths}
    volume_series: Dict[int, List[float]] = {cl: [] for cl in chunk_lengths}

    for idx, row in tqdm.tqdm(dataset.iterrows(),
                              desc='Generating numerical features',
                              total=len(dataset)):

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()
        os.makedirs(os.path.join(output_dir, 'bert_embeddings', str(hashed_dir)), exist_ok=True)

        embeddings_path = os.path.join(output_dir, 'bert_embeddings', str(hashed_dir), f"{idx}.pt")
        if os.path.exists(embeddings_path):
            sentence_embeddings = torch.load(embeddings_path)
        else:
            sentence_embeddings = text_processor.get_bert_embeddings(row['review_text'])
            torch.save(sentence_embeddings, embeddings_path)

        for chunk_length in chunk_lengths:
            trace_velocity = text_processor.calc_trace_velocity(
                sentence_embeddings,
                chunk_length=chunk_length
            )
            trace_volume = text_processor.calc_trace_volume(
                sentence_embeddings,
                chunk_length=chunk_length
            )

            velocity_series[chunk_length].append(trace_velocity)
            volume_series[chunk_length].append(trace_volume)

    features = pd.concat((
        dataset['restaurant_href'],
        dataset['review_text'].map(text_processor.num_words),
        dataset['review_text'].map(text_processor.num_sentences),
        *(pd.Series(velocity_series[cl]) for cl in chunk_lengths),
        *(pd.Series(volume_series[cl]) for cl in chunk_lengths),
    ),
        axis=1,
        keys=['restaurant_href', 'num_words', 'num_sentences',
              *(f'trace_velocity_cl_{cl}' for cl in chunk_lengths),
              *(f'trace_volume_cl_{cl}' for cl in chunk_lengths)]
    )

    features.to_pickle(os.path.join(output_dir, 'numerical_features.pkl'))


def _prepare_pos_based_features(dataset: pd.DataFrame,
                                text_processor: text_processing.TextPreprocessor,
                                output_dir: str):
    """Prepares and saves POS-based features for the dataset."""

    _logger().info('Preparing POS-based features...')

    for idx, row in tqdm.tqdm(dataset.iterrows(),
                              desc='Generating POS-based features',
                              total=len(dataset)):

        pos_bow_repr = text_processor.get_pos_bow_representation(row['review_text'])

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()
        os.makedirs(os.path.join(output_dir, str(hashed_dir)), exist_ok=True)

        torch.save(pos_bow_repr,
                   os.path.join(output_dir, str(hashed_dir), f'{idx}.pt'))


@hydra.main(version_base=None, config_path="cfg", config_name="process_dataset")
def main(cfg: omegaconf.DictConfig):
    """Loads and processes the dataset according to the provided configuration."""

    logging_utils.setup_logging('process_dataset')

    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_container(cfg))

    os.makedirs(cfg.output_path, exist_ok=True)

    _logger().info('Preparing preprocessed dataset...')

    text_processor = text_processing.TextPreprocessor(bert_model_device=cfg.bert_model_device)

    prep_ds = _obtain_preprocessed_ds(
        raw_dataset=raw_ds.RawDSLoader(cfg.raw_ds_path).load_dataset(),
        text_processor=text_processor,
        output_path=cfg.output_path
    )

    _logger().info('Preprocessed dataset contains %d reviews.', len(prep_ds))
    _logger().info('Building vocabulary for BOW representations...')

    text_processor.update_vocabulary(prep_ds['review_text'].tolist())

    text_processor.save_vocabulary_to_file(
        vocabulary=text_processor.vocabulary,
        filepath=os.path.join(cfg.output_path, "vocabulary.txt")
    )

    top_vocab, bottom_vocab = text_processor.top_bottom_n_words(cfg.vocabulary_top_bottom_words)

    for label, words in (("top", top_vocab), ("bottom", bottom_vocab)):

        vocab_path = os.path.join(cfg.output_path, f"vocabulary_{label}.txt")
        text_processing.TextPreprocessor.save_vocabulary_to_file(
            vocabulary=words,
            filepath=vocab_path
        )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, "vocabulary_top.txt"),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, "bow_representations_top"),
        use_tfidf=False
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, "vocabulary_bottom.txt"),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, "bow_representations_bottom"),
        use_tfidf=False
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, "vocabulary_top.txt"),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, "tfidf_representations_top"),
        use_tfidf=True
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, "vocabulary_bottom.txt"),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, "tfidf_representations_bottom"),
        use_tfidf=True
    )

    _prepare_numerical_features(
        dataset=prep_ds,
        text_processor=text_processor,
        output_dir=cfg.output_path
    )

    text_processor.update_pos_vocab(prep_ds['review_text'].tolist())

    text_processor.save_pos_vocab_to_file(
        filepath=os.path.join(cfg.output_path, "pos_vocabulary.txt")
    )

    _prepare_pos_based_features(
        dataset=prep_ds,
        text_processor=text_processor,
        output_dir=os.path.join(cfg.output_path, "pos_bow")
    )

    _logger().info('Saving dataset stats...')


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
