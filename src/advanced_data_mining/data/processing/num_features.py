"""Utilities for extracting numerical features from text data."""

import logging
import sys

import pydantic
import gruut
import nltk
import numpy as np
import torch


def num_words(text: str) -> int:
    """Returns the number of words in the given text."""
    words = nltk.tokenize.word_tokenize(text)
    return len(words)


def num_sentences(text: str) -> int:
    """Returns the number of sentences in the given text."""
    sentences = nltk.tokenize.sent_tokenize(text)
    return len(sentences)


def normalize_text(text: str) -> str:
    """Normalizes text.

    Normalization includes converting currency, numbers to words, and standardizing punctuation.
    """

    sentences = [sentence.text_with_ws for sentence in gruut.sentences(text, phonemes=False)]
    return ' '.join(sentences)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class NumericalFeaturesExtractorCfg(pydantic.BaseModel):
    """Configuration for NumericalFeaturesExtractor."""
    chunk_and_step_sizes: list[tuple[int, int]]


class NumericalFeaturesExtractor:
    """Preprocesses text data for further analysis."""

    def __init__(self, cfg: NumericalFeaturesExtractorCfg):

        self._cfg = cfg

    @property
    def cfg(self) -> NumericalFeaturesExtractorCfg:
        """Returns the configuration of the NumericalFeaturesExtractor."""
        return self._cfg

    def generate_trace_features(self,
                                word_embeddings: torch.Tensor
                                ) -> list[dict[str, float]]:
        """Generates (velocity, volume) for the given word embeddings for all chunking settings."""

        features: list[dict[str, float]] = []

        for chunk_length, step_size in self._cfg.chunk_and_step_sizes:
            chunks = self._prepare_embedding_chunks(word_embeddings, chunk_length, step_size)

            trace_velocity = self._calc_trace_velocity(chunks)
            trace_volume = self._calc_trace_volume(chunks)

            features.append({
                "chunk_length": chunk_length,
                "step_size": step_size,
                "trace_velocity": trace_velocity,
                "trace_volume": trace_volume
            })

        return features

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
