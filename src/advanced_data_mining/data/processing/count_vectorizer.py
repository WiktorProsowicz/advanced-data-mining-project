"""A module that generates basic linguistic representations leveraging the word stats."""

import pickle
import json

import pathlib
from typing import Iterable, Iterator
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
import pydantic

from advanced_data_mining.data.structs import raw_ds


class CountVectorizerConfig(pydantic.BaseModel):
    """Configuration for CountVectorizer."""

    max_vocabulary_size: int | None
    skip_stop_words: bool = True


def iterate_docs(dataset: raw_ds.RawDataset) -> Iterator[str]:
    """Yields documents from the dataset."""

    for reviews in dataset.values():
        for review in reviews:
            yield review.text


class CountVectorizer:
    """Generates vectorized document representations based on word occurrence statistics."""

    @classmethod
    def from_path(cls, path: pathlib.Path) -> 'CountVectorizer':
        """Loads a serialized CountVectorizer from the provided path."""

        doc_frequency_vector = np.load(path / 'doc_frequency_vector.npy')

        with path.joinpath('vectorizer.pkl').open('rb') as f:
            base_vectorizer = pickle.load(f)

        with path.joinpath('config.json').open('r') as f:
            cfg_dict = json.load(f)
            cfg = CountVectorizerConfig.model_validate(cfg_dict)

        return cls(cfg, base_vectorizer, doc_frequency_vector)

    def __init__(self,
                 cfg: CountVectorizerConfig,
                 base_vectorizer: SklearnCountVectorizer | None,
                 doc_frequency_vector: np.ndarray | None) -> None:

        nltk.download('stopwords', quiet=True)

        self._cfg = cfg

        if doc_frequency_vector is None:
            self._doc_frequency_vector = np.array([], dtype=np.int32)
        else:
            self._doc_frequency_vector = doc_frequency_vector

        if base_vectorizer is None:

            if cfg.skip_stop_words:
                stop_words = nltk.corpus.stopwords.words('english')
            else:
                stop_words = None

            self._base_vectorizer = SklearnCountVectorizer(
                max_features=cfg.max_vocabulary_size,
                stop_words=stop_words,
            )
        else:
            self._base_vectorizer = base_vectorizer

    def fit(self, docs: list[str]) -> None:
        """Fits the vectorizer to the provided documents."""

        self._base_vectorizer.fit(docs)

        self._doc_frequency_vector = np.zeros(
            len(self._base_vectorizer.vocabulary_), dtype=np.int32)

        for doc in docs:
            vector = self._base_vectorizer.transform([doc])
            self._doc_frequency_vector += (vector.toarray()[0] > 0).astype(np.int32)

    def transform(self, documents: Iterable[str]) -> np.ndarray:
        """Transforms the provided documents into vectorized representations."""

        return self._base_vectorizer.transform(documents).toarray()  # type: ignore

    def serialize(self, path: pathlib.Path) -> None:
        """Serializes the vectorizer to the provided path."""

        np.save(path / 'doc_frequency_vector.npy', self._doc_frequency_vector)

        with path.joinpath('vectorizer.pkl').open('wb') as f:
            pickle.dump(self._base_vectorizer, f)

        with path.joinpath('config.json').open('w') as f:
            json.dump(self._cfg.model_dump(), f)
