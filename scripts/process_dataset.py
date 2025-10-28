# -*- coding: utf-8 -*-
"""Transforms raw dataset into a processed format suitable for model training.

The script performs the following steps:
1. Normalization of review texts.
2. Vocabulary building from the normalized texts.
3. Generation of Bag-of-Words (BOW) representations for each review based on top/bottom N words
    from the vocabulary.
4. Generation of basic numerical features for each review.
5. Generation of features derived from advanced text representations (e.g., BERT embeddings).
"""
import hashlib
import json
import logging
import os
from typing import Dict
from typing import List
from typing import Tuple

import hydra
import omegaconf
import pandas as pd  # type: ignore
import torch
import tqdm  # type: ignore

from advanced_data_mining.data import raw_ds
from advanced_data_mining.data import text_processing
from advanced_data_mining.utils import logging_utils


def _logger():
    return logging.getLogger(__name__)


def _obtain_preprocessed_ds(raw_dataset: raw_ds.RawDataset,
                            text_processor: text_processing.TextPreprocessor,
                            output_path: str) -> pd.DataFrame:

    preprocessed_ds_path = os.path.join(output_path, 'preprocessed_dataset.pkl')

    if os.path.exists(preprocessed_ds_path):
        return pd.read_pickle(preprocessed_ds_path)

    preprocessed_reviews = []

    for restaurant, reviews in tqdm.tqdm(raw_dataset.items(),
                                         desc='Processing restaurants',
                                         total=len(raw_dataset)):

        for review in tqdm.tqdm(reviews,
                                desc='Processing reviews',
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
                 restaurant.city == 'Krakow',
                 preprocessed_text,
                 review.rating)
            )

    prep_ds = pd.DataFrame(preprocessed_reviews,
                           columns=['restaurant_href', 'restaurant_name', 'restaurant_basic_info',
                                    'is_from_cracow', 'review_text', 'review_rating'])

    prep_ds.to_pickle(preprocessed_ds_path)

    return prep_ds


def _prepare_bow_representations(vocabulary_path: str,
                                 dataset: pd.DataFrame,
                                 output_path: str,
                                 use_tfidf: bool):
    """Prepares and saves BOW representations for the dataset."""

    text_processor = text_processing.TextPreprocessor()
    text_processor.load_vocab_from_file(vocabulary_path)

    _logger().info('Saving BOW representations to %s...', output_path)

    os.makedirs(output_path, exist_ok=True)

    for idx, row in tqdm.tqdm(dataset.iterrows(),
                              desc='Generating BOW representations',
                              total=len(dataset)):
        if use_tfidf:
            bow_repr = text_processor.get_tfidf_representation(row['review_text'])
        else:
            bow_repr = text_processor.get_bow_representation(row['review_text'])

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()
        os.makedirs(os.path.join(output_path, str(hashed_dir)), exist_ok=True)

        torch.save(bow_repr,
                   os.path.join(output_path, str(hashed_dir), f'{idx}.pt'))


def _prepare_bert_embeddings(dataset: pd.DataFrame,
                             text_processor: text_processing.TextPreprocessor,
                             output_dir: str):
    """Prepares and saves BERT embeddings for the dataset."""

    _logger().info('Preparing BERT embeddings...')

    for idx, row in tqdm.tqdm(dataset.iterrows(),
                              desc='Generating BERT embeddings',
                              total=len(dataset)):

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()

        word_embs_path = os.path.join(output_dir,
                                      'word_bert_embeddings',
                                      str(hashed_dir),
                                      f'{idx}.pt')
        sentence_embs_path = os.path.join(output_dir,
                                          'sentence_bert_embeddings',
                                          str(hashed_dir),
                                          f'{idx}.pt')

        os.makedirs(os.path.dirname(word_embs_path), exist_ok=True)
        os.makedirs(os.path.dirname(sentence_embs_path), exist_ok=True)

        word_embs, sentence_embs = text_processor.get_bert_embeddings(row['review_text'])

        torch.save(word_embs, word_embs_path)
        torch.save(sentence_embs, sentence_embs_path)


def _prepare_numerical_features(dataset: pd.DataFrame,
                                text_processor: text_processing.TextPreprocessor,
                                chunking_cfg: List[Dict[str, int]],
                                output_dir: str):
    """Prepares and saves numerical features for the dataset."""

    _logger().info('Preparing numerical features...')

    chunks_data = [(cfg['chunk_length'], cfg['step_size']) for cfg in chunking_cfg]

    velocity_series: Dict[Tuple[int, int], List[float]] = {(cl, sz): [] for cl, sz in chunks_data}
    volume_series: Dict[Tuple[int, int], List[float]] = {(cl, sz): [] for cl, sz in chunks_data}

    for idx, row in tqdm.tqdm(dataset.iterrows(),
                              desc='Generating numerical features',
                              total=len(dataset)):

        hashed_dir = hashlib.sha256(bytes(row['restaurant_href'],
                                          encoding='utf-8')).hexdigest()

        embeddings = torch.load(os.path.join(output_dir,
                                             'word_bert_embeddings',
                                             str(hashed_dir),
                                             f'{idx}.pt'))

        for ch_len, step_size in chunks_data:
            trace_velocity = text_processor.calc_trace_velocity(
                embeddings,
                chunk_length=ch_len,
                step_size=step_size
            )
            trace_volume = text_processor.calc_trace_volume(
                embeddings,
                chunk_length=ch_len,
                step_size=step_size
            )

            velocity_series[(ch_len, step_size)].append(trace_velocity)
            volume_series[(ch_len, step_size)].append(trace_volume)

    features = pd.concat((
        dataset['restaurant_href'],
        dataset['review_text'].map(text_processor.num_words).rename('num_words'),
        dataset['review_text'].map(text_processor.num_sentences).rename('num_sentences'),
        dataset['review_rating'],
        dataset['is_from_cracow'],
        *(pd.Series(velocity_series[(cl, sz)],
                    name=f'trace_velocity_cl_{cl}_sz_{sz}')
          for cl, sz in chunks_data),
        *(pd.Series(volume_series[(cl, sz)],
                    name=f'trace_volume_cl_{cl}_sz_{sz}')
          for cl, sz in chunks_data),
    ),
        axis=1
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


@hydra.main(version_base=None, config_path='cfg', config_name='process_dataset')
def main(cfg: omegaconf.DictConfig):
    """Loads and processes the dataset according to the provided configuration."""

    logging_utils.setup_logging('process_dataset')

    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_container(cfg))

    os.makedirs(cfg.output_path, exist_ok=True)

    with open(os.path.join(cfg.output_path, 'metadata.json'), 'w', encoding='utf-8') as meta_f:
        json.dump(omegaconf.OmegaConf.to_container(cfg), meta_f, indent=4)

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
        filepath=os.path.join(cfg.output_path, 'vocabulary.txt')
    )

    top_vocab, bottom_vocab = text_processor.top_bottom_n_words(
        n_top=cfg.top_words_for_bow_repr,
        n_bottom=cfg.bottom_words_for_bow_repr
    )

    for label, words in (('top', top_vocab), ('bottom', bottom_vocab)):

        vocab_path = os.path.join(cfg.output_path, f'vocabulary_{label}.txt')
        text_processing.TextPreprocessor.save_vocabulary_to_file(
            vocabulary=words,
            filepath=vocab_path
        )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary_top.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'bow_representations_top'),
        use_tfidf=False
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary_bottom.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'bow_representations_bottom'),
        use_tfidf=False
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'bow_representations_full'),
        use_tfidf=False
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary_top.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'tfidf_representations_top'),
        use_tfidf=True
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary_bottom.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'tfidf_representations_bottom'),
        use_tfidf=True
    )

    _prepare_bow_representations(
        vocabulary_path=os.path.join(cfg.output_path, 'vocabulary.txt'),
        dataset=prep_ds,
        output_path=os.path.join(cfg.output_path, 'tfidf_representations_full'),
        use_tfidf=True
    )

    _prepare_bert_embeddings(
        dataset=prep_ds,
        text_processor=text_processor,
        output_dir=cfg.output_path
    )

    _prepare_numerical_features(
        dataset=prep_ds,
        text_processor=text_processor,
        chunking_cfg=omegaconf.OmegaConf.to_container(cfg.volume_velocity_cfg),
        output_dir=cfg.output_path
    )

    text_processor.update_pos_vocab(prep_ds['review_text'].tolist())

    text_processor.save_pos_vocab_to_file(
        filepath=os.path.join(cfg.output_path, 'pos_vocabulary.txt')
    )

    _prepare_pos_based_features(
        dataset=prep_ds,
        text_processor=text_processor,
        output_dir=os.path.join(cfg.output_path, 'pos_bow')
    )

    _logger().info('Saving dataset stats...')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
