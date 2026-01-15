"""Contains definition preprocessed dataset loader."""
import logging
import os
import random
import re
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import lightning as pl
import pandas as pd
import torch

from advanced_data_mining.utils import misc


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ProcessedDataset(torch.utils.data.Dataset[Dict[str, torch.Tensor]]):
    """Reads and loads preprocessed dataset samples."""

    def __init__(self, ds_path: str, sample_indices: List[int]):

        self._ds_path = ds_path

        if not os.path.exists(self._ds_path):
            _logger().error('Dataset path %s does not exist.', self._ds_path)
            sys.exit(1)

        self._numerical_features: pd.DataFrame = pd.read_pickle(
            os.path.join(self._ds_path, 'numerical_features.pkl'))

        self._preprocessed_ds: pd.DataFrame = pd.read_pickle(
            os.path.join(self._ds_path, 'preprocessed_dataset.pkl')
        )

        self._sample_indices = sample_indices

    def __len__(self) -> int:
        return len(self._sample_indices)

    def get_raw_sample(self, idx: int) -> Dict[str, Any]:
        """Returns unprocessed data for a given sample."""

        sample_idx = self._sample_indices[idx]

        return {str(k): v
                for k, v in self._preprocessed_ds.iloc[sample_idx].to_dict().items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample_idx = self._sample_indices[idx]
        num_features = self._numerical_features.iloc[sample_idx]

        restaurant_hash = misc.hash_restaurant_href(num_features['restaurant_href'])

        bow_bottom_path = os.path.join(self._ds_path,
                                       'bow_representations_bottom',
                                       restaurant_hash,
                                       f'{sample_idx}.pt')

        bow_top_path = os.path.join(self._ds_path,
                                    'bow_representations_top',
                                    restaurant_hash,
                                    f'{sample_idx}.pt')

        pos_bow_path = os.path.join(self._ds_path,
                                    'pos_bow',
                                    restaurant_hash,
                                    f'{sample_idx}.pt')

        tfidf_bottom_path = os.path.join(self._ds_path,
                                         'tfidf_representations_bottom',
                                         restaurant_hash,
                                         f'{sample_idx}.pt')

        tfidf_top_path = os.path.join(self._ds_path,
                                      'tfidf_representations_top',
                                      restaurant_hash,
                                      f'{sample_idx}.pt')

        bow_full_path = os.path.join(self._ds_path,
                                     'bow_representations_full',
                                     restaurant_hash,
                                     f'{sample_idx}.pt')

        tfidf_full_path = os.path.join(self._ds_path,
                                       'tfidf_representations_full',
                                       restaurant_hash,
                                       f'{sample_idx}.pt')
        sentence_bert_path = os.path.join(self._ds_path,
                                          'sentence_bert_embeddings',
                                          restaurant_hash,
                                          f'{sample_idx}.pt')

        trace_cols = [col for col in self._numerical_features.columns
                      if col.startswith('trace_')]

        trace_re = re.compile(r'cl_(\d+)_sz_(\d+)')

        trace_features_specs = set()

        for col in trace_cols:
            match = trace_re.search(col)
            if match:
                trace_features_specs.add(match.group(0))

        trace_features = {
            spec: torch.tensor([num_features[f'trace_velocity_{spec}'],
                                num_features[f'trace_volume_{spec}']], dtype=torch.float32)
            for spec in trace_features_specs
        }

        return {
            'bow_bottom': torch.load(bow_bottom_path).to_dense(),
            'bow_top': torch.load(bow_top_path).to_dense(),
            'pos_bow': torch.load(pos_bow_path).to_dense().to(torch.float32),
            'tfidf_bottom': torch.load(tfidf_bottom_path).to_dense(),
            'tfidf_top': torch.load(tfidf_top_path).to_dense(),
            'bow_full': torch.load(bow_full_path).to_dense(),
            'tfidf_full': torch.load(tfidf_full_path).to_dense(),
            'sentence_bert': torch.load(sentence_bert_path).to(torch.float32),
            'is_from_cracow': torch.tensor(num_features['is_from_cracow'],
                                           dtype=torch.float32).unsqueeze(-1),
            'num_words': torch.tensor(num_features['num_words'],
                                      dtype=torch.float32).unsqueeze(-1),
            'num_sentences': torch.tensor(num_features['num_sentences'],
                                          dtype=torch.float32).unsqueeze(-1),
            'review_rating': torch.tensor(num_features['review_rating'],
                                          dtype=torch.float32),
            **trace_features
        }


class ProcessedDataModule(pl.LightningDataModule):
    """Lightning data module for the processed dataset."""

    def __init__(self,
                 ds_path: str,
                 batch_size: int,
                 n_workers: int,
                 n_test_samples: int,
                 train_val_split: float):
        super().__init__()
        self._ds_path = ds_path
        self._batch_size = batch_size
        self._n_test_samples = n_test_samples
        self._train_val_split = train_val_split
        self._n_workers = n_workers

        self._train_ds: Optional[ProcessedDataset] = None
        self._val_ds: Optional[ProcessedDataset] = None
        self._test_ds: Optional[ProcessedDataset] = None

    @property
    def test_ds(self) -> ProcessedDataset:
        """Returns the test dataset."""
        assert self._test_ds is not None, 'The test dataset has not been set up yet.'
        return self._test_ds

    def setup(self, _: str) -> None:
        """Sets up the datasets for training, validation and testing."""

        preprocessed_ds = pd.read_pickle(
            os.path.join(self._ds_path, 'preprocessed_dataset.pkl')
        )

        all_indices = preprocessed_ds.index.tolist()

        random.shuffle(all_indices)

        test_indices = all_indices[-self._n_test_samples:]
        train_val_indices = all_indices[:-self._n_test_samples]

        n_train = int(len(train_val_indices) * self._train_val_split)
        train_indices = train_val_indices[:n_train]
        val_indices = train_val_indices[n_train:]

        self._train_ds = ProcessedDataset(ds_path=self._ds_path,
                                          sample_indices=train_indices)
        self._val_ds = ProcessedDataset(ds_path=self._ds_path,
                                        sample_indices=val_indices)
        self._test_ds = ProcessedDataset(ds_path=self._ds_path,
                                         sample_indices=test_indices)

    def train_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, torch.Tensor]]:
        """Returns the training data loader."""

        assert self._train_ds is not None, 'The training dataset has not been set up yet.'

        return torch.utils.data.DataLoader(self._train_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, torch.Tensor]]:
        """Returns the validation data loader."""

        assert self._val_ds is not None, 'The validation dataset has not been set up yet.'

        return torch.utils.data.DataLoader(self._val_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, torch.Tensor]]:
        """Returns the test data loader."""

        assert self._test_ds is not None, 'The test dataset has not been set up yet.'

        return torch.utils.data.DataLoader(self._test_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=False)
