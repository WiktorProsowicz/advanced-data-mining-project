"""Contains definition preprocessed dataset loader."""
import json
import logging
from typing import Literal, Annotated
import dataclasses
import pathlib

import pydantic
from pydantic import Field
import torch
import numpy as np
import lightning.pytorch as pl

from advanced_data_mining.data.structs import raw_ds, processed_ds


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclasses.dataclass
class SampleMetadata:
    """Holds metadata about a processed review sample."""
    review: raw_ds.Review
    restaurant_info: raw_ds.Restaurant
    n_author_reviews_index: int
    is_translated: bool
    n_words: int
    n_sentences: int


class ProcessedDatasetConfig(pydantic.BaseModel):
    """Configuration for the processed dataset loader."""

    use_bert_embeddings: Annotated[Literal['sentence', 'review'] | None, Field(
        description=('Whether to load all BERT embeddings for each sentence, or to load only the'
                     'average BERT embedding for the whole review. If None, no BERT embeddings'
                     'will be loaded.')
    )] = None

    word_count_vector_type: Annotated[Literal['binary', 'count',
                                              'tfidf', 'count_normalized'] | None, Field(
        description=('Type of word count vector to load. Options are:'
                     'binary - presence/absence of words;'
                     'count - raw word counts;'
                     'tfidf - term frequency-inverse document frequency;'
                     'count_normalized - word counts normalized to [0, 1] scale.')
    )] = None

    use_top_k_words: Annotated[int | None, Field(
        description=('If set, only the top K most frequent words will be used as features. If None,'
                     'all words will be used.')
    )] = None

    pos_count_vector_type: Annotated[Literal['count', 'count_normalized'] | None, Field(
        description=('Type of part-of-speech count vector to load. Options are:'
                     'count - raw counts of each part of speech;'
                     'count_normalized - counts normalized to [0, 1] scale.)')
    )] = None

    use_categorized_features: Annotated[list[str] | None, Field(
        description=('List of categorized features to load. If None, all categorized features will'
                     'be loaded.')
    )] = None

    use_trace_features: Annotated[list[tuple[int, int]] | None, Field(
        description=('List of trace features to load, specified as tuples of'
                     '(chunk_size, step_size). If None, all trace features will be loaded.')
    )] = None

    normalize_trace_features: Annotated[bool, Field(
        description=('Whether to normalize trace features to 0 mean and 1 variance.')
    )] = False


class ProcessedDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Reads and loads preprocessed dataset samples."""

    def __init__(self,
                 cfg: ProcessedDatasetConfig,
                 metadata_handler: processed_ds.ProcessingMetadataPathHandler,
                 samples: list[processed_ds.ProcessedReview]) -> None:

        self._cfg = cfg
        self._metadata_handler = metadata_handler
        self._samples = samples

        doc_freq_pth = self._metadata_handler.count_vectorizer_path / 'doc_frequency_vector.npy'
        word_count_scaling_pth = (self._metadata_handler.scaling_metadata_path /
                                  'count_vectors_scale.pt')

        self._doc_frequency_vector = torch.from_numpy(np.load(doc_freq_pth)).to(torch.float32)
        self._word_count_scale = torch.load(word_count_scaling_pth).to(torch.float32)
        self._pos_count_scale = torch.load(self._metadata_handler.scaling_metadata_path /
                                           'pos_vectors_scale.pt').to(torch.float32)

        with (self._metadata_handler.count_vectorizer_path / 'documents_count').open('r') as f:
            self._documents_count = int(f.read())

        self._trace_scaling_params = self._metadata_handler.get_trace_scaling_params()

        self._top_k_indices: torch.Tensor | None = None

        if self._cfg.use_top_k_words is not None:
            self._top_k_indices = torch.topk(self._doc_frequency_vector,
                                             self._cfg.use_top_k_words).indices
            self._doc_frequency_vector = self._doc_frequency_vector[self._top_k_indices]
            self._word_count_scale = self._word_count_scale[self._top_k_indices]

    def __len__(self) -> int:
        return len(self._samples)

    def get_raw_sample(self, idx: int) -> SampleMetadata:
        """Returns unprocessed data for a given sample."""

        sample = self._samples[idx]

        with sample.num_features_pth.open('r', encoding='utf-8') as f:
            num_features = json.load(f)

        return SampleMetadata(
            review=sample.raw_review,
            restaurant_info=sample.restaurant_info,
            n_author_reviews_index=num_features['n_author_reviews_index'],
            is_translated=num_features['is_translated'],
            n_sentences=num_features['n_sentences'],
            n_words=num_features['n_words']
        )

    def collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Custom collate function to handle variable-length features."""

        collated_batch: dict[str, torch.Tensor] = {}

        for feat_name in self._supported_categorized_features() + ['rating']:
            collated_batch[feat_name] = torch.stack([sample[feat_name] for sample in batch])

        for cs, ss in self._supported_trace_features():
            collated_batch[f'trace_cs_{cs}_ss_{ss}'] = torch.stack(
                [sample[f'trace_cs_{cs}_ss_{ss}'] for sample in batch]
            )

        if self._cfg.word_count_vector_type is not None:
            collated_batch['word_count_vector'] = torch.stack([sample['word_count_vector']
                                                               for sample in batch])

        if self._cfg.use_bert_embeddings is not None:

            if self._cfg.use_bert_embeddings == 'sentence':
                collated_batch['bert_embeddings'] = torch.nn.utils.rnn.pad_sequence(
                    [sample['bert_embeddings'] for sample in batch],
                    batch_first=True,
                    padding_value=0.0
                )
                collated_batch['n_sentences'] = torch.tensor([sample['n_sentences']
                                                              for sample in batch])

            else:
                collated_batch['bert_embeddings'] = torch.stack([sample['bert_embeddings']
                                                                 for sample in batch])

        if self._cfg.pos_count_vector_type is not None:
            collated_batch['pos_count_vector'] = torch.stack([sample['pos_count_vector']
                                                              for sample in batch])

        return collated_batch

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        sample = self._samples[idx]

        data: dict[str, torch.Tensor] = {}

        if self._cfg.use_bert_embeddings is not None:

            data['bert_embeddings'] = torch.load(sample.bert_embeddings_pth)

            if self._cfg.use_bert_embeddings == 'review':
                data['bert_embeddings'] = data['bert_embeddings'].mean(dim=0)

            else:
                data['n_sentences'] = torch.tensor(data['bert_embeddings'].size(0))

        data.update(self._load_word_count_features(sample))

        if self._cfg.pos_count_vector_type is not None:
            data['pos_count_vector'] = torch.load(sample.pos_count_vector_pth).to(torch.float32)

            if self._cfg.pos_count_vector_type == 'count_normalized':
                data['pos_count_vector'] = torch.clip(data['pos_count_vector'] /
                                                      self._pos_count_scale, 0, 1)

        supported_cat_features = self._supported_categorized_features()

        with sample.num_features_pth.open('r', encoding='utf-8') as f:
            num_features = json.load(f)

            data['rating'] = torch.tensor(num_features['rating'])
            data['is_translated'] = torch.tensor(num_features['is_translated'], dtype=torch.bool)
            cat_features = num_features['encoded_cat_options']

        for cat_feature_name in supported_cat_features:
            data[cat_feature_name] = torch.tensor(cat_features[cat_feature_name])

        data.update(self._load_trace_features(sample))

        return data

    def _supported_trace_features(self) -> list[tuple[int, int]]:
        """Returns the list of supported trace features (chunk_size, step_size)."""

        with self._metadata_handler.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        all_features = [tuple(chunk_setup) for chunk_setup in cfg['chunk_and_step_sizes']]

        if self._cfg.use_trace_features is not None:
            return [feat for feat in all_features if feat in self._cfg.use_trace_features]

        return all_features

    def _supported_categorized_features(self) -> list[str]:
        """Returns the list of supported categorized features."""

        with self._metadata_handler.numerical_features_cfg_path.open('r', encoding='utf-8') as f:
            cfg = json.load(f)

        all_features = [option_setup['name'] for option_setup in
                        cfg['categorized_options_used']]

        if self._cfg.use_categorized_features is not None:
            return [feat for feat in all_features if feat in self._cfg.use_categorized_features]

        return all_features

    def _load_trace_features(self, sample: processed_ds.ProcessedReview) -> dict[str, torch.Tensor]:
        """Loads trace features for a given review."""

        data: dict[str, torch.Tensor] = {}

        supported_features = self._supported_trace_features()

        with sample.trace_features_pth.open('r', encoding='utf-8') as f:
            trace_features = json.load(f)

        for trace_feature_spec in trace_features:

            chunk_size = trace_feature_spec['chunk_length']
            step_size = trace_feature_spec['step_size']

            if (chunk_size, step_size) not in supported_features:
                continue

            feature_tensor = torch.tensor([trace_feature_spec['trace_velocity'],
                                           trace_feature_spec['trace_volume']])

            if self._cfg.normalize_trace_features:
                mean, std = self._trace_scaling_params[(chunk_size, step_size)]
                feature_tensor = (feature_tensor - mean) / std

            data[f'trace_cs_{chunk_size}_ss_{step_size}'] = feature_tensor

        return data

    def _load_word_count_features(self,
                                  sample: processed_ds.ProcessedReview) -> dict[str, torch.Tensor]:
        """Loads word count vector features for a given sample."""

        if self._cfg.word_count_vector_type is None:
            return {}

        word_count_vector = torch.load(sample.word_count_vector_pth).to_dense().to(torch.float32)

        if self._top_k_indices is not None:
            word_count_vector = word_count_vector[self._top_k_indices]

        if self._cfg.word_count_vector_type == 'count_normalized':
            word_count_vector = torch.clip(word_count_vector / self._word_count_scale, 0, 1)

        if self._cfg.word_count_vector_type == 'tfidf':
            tf = word_count_vector / (word_count_vector.sum() + 1e-8)
            idf = torch.log((1 + self._documents_count) /
                            (1 + self._doc_frequency_vector)) + 1.0

            word_count_vector = tf * idf

        return {
            'word_count_vector': word_count_vector
        }


class ProcessedDataModule(pl.LightningDataModule):
    """LightningDataModule for the processed dataset."""

    def __init__(self,
                 ds_cfg: ProcessedDatasetConfig,
                 ds_path: pathlib.Path,
                 metadata_path: pathlib.Path,
                 batch_size: int,
                 n_workers: int,
                 train_val_split: float):
        super().__init__()

        self._ds_path_handler = processed_ds.ProcessedDsPathHandler(ds_path)
        self._metadata_handler = processed_ds.ProcessingMetadataPathHandler(metadata_path)
        self._ds_cfg = ds_cfg
        self._batch_size = batch_size
        self._n_workers = n_workers
        self._train_val_split = train_val_split

        self._train_dataset: ProcessedDataset | None = None
        self._val_dataset: ProcessedDataset | None = None
        self._test_dataset: ProcessedDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Initializes datasets for the specified stage (fit or test)."""

        all_samples = list(self._ds_path_handler.iter_all_reviews())

        if stage in (None, 'fit'):
            split_idx = int(len(all_samples) * self._train_val_split)
            train_samples = all_samples[:split_idx]
            val_samples = all_samples[split_idx:]

            self._train_dataset = ProcessedDataset(self._ds_cfg,
                                                   self._metadata_handler,
                                                   train_samples)
            self._val_dataset = ProcessedDataset(self._ds_cfg,
                                                 self._metadata_handler,
                                                 val_samples)

        if stage in (None, 'test'):
            self._test_dataset = ProcessedDataset(self._ds_cfg,
                                                  self._metadata_handler,
                                                  all_samples)

    def train_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        """Returns DataLoader for the training set."""

        assert self._train_dataset is not None, 'Train dataset not initialized'

        return torch.utils.data.DataLoader(self._train_dataset,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           shuffle=True,
                                           collate_fn=self._train_dataset.collate_fn)

    def val_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        """Returns DataLoader for the validation set."""

        assert self._val_dataset is not None, 'Validation dataset not initialized'

        return torch.utils.data.DataLoader(self._val_dataset,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           shuffle=False,
                                           collate_fn=self._val_dataset.collate_fn)

    def test_dataloader(self) -> torch.utils.data.DataLoader[dict[str, torch.Tensor]]:
        """Returns DataLoader for the test set."""

        assert self._test_dataset is not None, 'Test dataset not initialized'

        return torch.utils.data.DataLoader(self._test_dataset,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           shuffle=False,
                                           collate_fn=self._test_dataset.collate_fn)
