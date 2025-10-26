# -*- coding: utf-8 -*-
"""Contains definition preprocessed dataset loader."""

from typing import Dict, List, Optional
import os
import logging
import sys
import random

import torch
import pandas as pd  # type: ignore
import lightning as pl

from advanced_data_mining.utils import misc


def _logger():
    return logging.getLogger(__name__)


class ProcessedDataset(torch.utils.data.Dataset):
    """Reads and loads preprocessed dataset samples."""

    def __init__(self, ds_path: str, sample_indices: List[int]):

        self._ds_path = ds_path

        if not os.path.exists(self._ds_path):
            _logger().error("Dataset path %s does not exist.", self._ds_path)
            sys.exit(1)

        self._numerical_features = pd.read_pickle(
            os.path.join(self._ds_path, 'numerical_features.pkl'))

        self._preprocessed_ds = pd.read_pickle(
            os.path.join(self._ds_path, 'preprocessed_dataset.pkl')
        )

        self._sample_indices = sample_indices

    def __len__(self) -> int:
        return len(self._sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        num_features = self._numerical_features.iloc[self._sample_indices[idx]]
        num_features['is_from_cracow'] = float(num_features['is_from_cracow'])

        restaurant_hash = misc.hash_restaurant_href(num_features['restaurant_href'])

        bow_bottom_path = os.path.join(self._ds_path,
                                       'bow_representations_bottom',
                                       restaurant_hash,
                                       f'{idx}.pt')

        bow_top_path = os.path.join(self._ds_path,
                                    'bow_representations_top',
                                    restaurant_hash,
                                    f'{idx}.pt')

        pos_bow_path = os.path.join(self._ds_path,
                                    'pos_bow',
                                    restaurant_hash,
                                    f'{idx}.pt')

        tfidf_bottom_path = os.path.join(self._ds_path,
                                         'tfidf_representations_bottom',
                                         restaurant_hash,
                                         f'{idx}.pt')

        tfidf_top_path = os.path.join(self._ds_path,
                                      'tfidf_representations_top',
                                      restaurant_hash,
                                      f'{idx}.pt')

        trace_features = {
            col: torch.tensor(num_features[col], dtype=torch.float)
            for col in num_features.index if col.startswith('trace_')
        }

        return {
            'bow_bottom': torch.load(bow_bottom_path).to_dense(),
            'bow_top': torch.load(bow_top_path).to_dense(),
            'pos_bow': torch.load(pos_bow_path).to_dense(),
            'tfidf_bottom': torch.load(tfidf_bottom_path).to_dense(),
            'tfidf_top': torch.load(tfidf_top_path).to_dense(),
            'is_from_cracow': torch.tensor(num_features['is_from_cracow'], dtype=torch.float),
            'num_words': torch.tensor(num_features['num_words'], dtype=torch.float),
            'num_sentences': torch.tensor(num_features['num_sentences'], dtype=torch.float),
            'review_rating': torch.tensor(num_features['review_rating'], dtype=torch.float),
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

    def setup(self, stage: str):  # pylint: disable=unused-argument
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

    def train_dataloader(self):
        """Returns the training data loader."""

        assert self._train_ds is not None, "The training dataset has not been set up yet."

        return torch.utils.data.DataLoader(self._train_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=True)

    def val_dataloader(self):
        """Returns the validation data loader."""

        assert self._val_ds is not None, "The validation dataset has not been set up yet."

        return torch.utils.data.DataLoader(self._val_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=False)

    def test_dataloader(self):
        """Returns the test data loader."""

        assert self._test_ds is not None, "The test dataset has not been set up yet."

        return torch.utils.data.DataLoader(self._test_ds,
                                           batch_size=self._batch_size,
                                           num_workers=self._n_workers,
                                           pin_memory=True,
                                           shuffle=False)
