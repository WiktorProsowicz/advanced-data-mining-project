"""Utilities for extracting numerical features from text data."""
import json
import logging
import pathlib
from typing import Any

import nltk
import numpy as np
import pydantic
import torch

from advanced_data_mining.data.structs import processed_ds


def num_words(text: str) -> int:
    """Returns the number of words in the given text."""
    words = nltk.tokenize.word_tokenize(text)
    return len(words)


def num_sentences(text: str) -> int:
    """Returns the number of sentences in the given text."""
    sentences = nltk.tokenize.sent_tokenize(text)
    return len(sentences)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class CategorizedOptionSetup(pydantic.BaseModel):
    """Configuration for processing of a categorized option."""
    name: str
    supported_values: list[str]


class NAuthorReviewsBinSetup(pydantic.BaseModel):
    """Configuration for processing number of author reviews feature."""
    cat_name: str
    min_val: int
    max_val: int


class NumericalFeaturesExtractorCfg(pydantic.BaseModel):
    """Configuration for NumericalFeaturesExtractor."""
    chunk_and_step_sizes: list[tuple[int, int]]
    categorized_options_used: list[CategorizedOptionSetup]
    n_author_reviews_quantization: list[NAuthorReviewsBinSetup]


class NumericalFeaturesExtractor:
    """Preprocesses text data for further analysis."""

    @classmethod
    def from_path(cls, path: pathlib.Path) -> 'NumericalFeaturesExtractor':
        """Creates a NumericalFeaturesExtractor from pre-serialized configuration."""

        with path.joinpath('config.json').open('r') as f:
            cfg_dict = json.load(f)
            cfg = NumericalFeaturesExtractorCfg.model_validate(cfg_dict)

        return cls(cfg)

    def __init__(self, cfg: NumericalFeaturesExtractorCfg) -> None:

        self._cfg = cfg

    @property
    def cfg(self) -> NumericalFeaturesExtractorCfg:
        """Returns the configuration of the NumericalFeaturesExtractor."""
        return self._cfg

    def generate_cat_options_onehot_indices(self,
                                            categorized_options: dict[str, str]
                                            ) -> dict[str, int]:
        """Generates one-hot indices for the given categorized options.

        If an option is missing, its index is 0. If an option has an unsupported value,
        its index is len(supported_values) + 1.
        """

        features: dict[str, int] = {}

        for option_setup in self._cfg.categorized_options_used:
            if option_setup.name not in categorized_options:
                features[option_setup.name] = 0
                continue

            if categorized_options[option_setup.name] not in option_setup.supported_values:
                features[option_setup.name] = len(option_setup.supported_values) + 1
                continue

            features[option_setup.name] = option_setup.supported_values.index(
                categorized_options[option_setup.name]) + 1

        return features

    def generate_n_author_reviews_onehot_index(self,
                                               n_author_reviews: int | None
                                               ) -> int:
        """Generates one-hot index for the given number of author reviews."""

        if n_author_reviews is None:
            return 0

        for idx, bin_setup in enumerate(self._cfg.n_author_reviews_quantization):
            if bin_setup.min_val <= n_author_reviews <= bin_setup.max_val:
                return idx + 1

        return len(self._cfg.n_author_reviews_quantization) + 1

    def generate_trace_features(self,
                                word_embeddings: torch.Tensor
                                ) -> list[dict[str, Any]]:
        """Generates (velocity, volume) for the given word embeddings for all chunking settings."""

        features: list[dict[str, float]] = []

        for chunk_length, step_size in self._cfg.chunk_and_step_sizes:
            chunks = self._prepare_embedding_chunks(word_embeddings, chunk_length, step_size)

            trace_velocity = self._calc_trace_velocity(chunks)
            trace_volume = self._calc_trace_volume(chunks)

            features.append({
                'chunk_length': chunk_length,
                'step_size': step_size,
                'trace_velocity': trace_velocity,
                'trace_volume': trace_volume
            })

        return features

    def serialize(self, output_dir: pathlib.Path) -> None:
        """Serializes the extractor configuration to the specified directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        with output_dir.joinpath('config.json').open('w', encoding='utf-8') as f:
            json.dump(self._cfg.model_dump(), f, ensure_ascii=False, indent=4)

    def _calc_trace_velocity(self,
                             chunks: list[torch.Tensor]) -> float:
        """Calculates trace velocity for the given sentence embeddings."""

        if len(chunks) < 2:
            return 0.0

        centroids = [chunk.mean(dim=0) for chunk in chunks]

        velocities = [torch.norm(centroids[i] - centroids[i - 1]).item()
                      for i in range(1, len(centroids))]

        return float(np.mean(velocities).item())

    def _calc_trace_volume(self,
                           chunks: list[torch.Tensor]) -> float:
        """Calculates trace volume for the given word embeddings."""

        if len(chunks) < 2:
            return 0.0

        centroids = torch.stack([chunk.mean(dim=0) for chunk in chunks])

        min_values = torch.min(centroids, dim=0).values
        max_values = torch.max(centroids, dim=0).values

        return float(torch.norm(max_values - min_values).item())

    def _prepare_embedding_chunks(self,
                                  word_embeddings: torch.Tensor,
                                  chunk_length: int,
                                  step_size: int) -> list[torch.Tensor]:
        """Prepares overlapping chunks from all given word embeddings."""

        return [word_embeddings[i:i + chunk_length]
                for i in range(0, word_embeddings.size(0) - chunk_length + 1, step_size)]
