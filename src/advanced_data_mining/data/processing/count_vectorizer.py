"""A module that generates basic linguistic representations leveraging the word stats."""

import pickle
import json

import pathlib
from typing import Iterable, Iterator
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer as SklearnCountVectorizer
import pydantic
import torch

from advanced_data_mining.data.structs import raw_ds


class CountVectorizerConfig(pydantic.BaseModel):
    """Configuration for CountVectorizer."""

    max_vocabulary_size: int | None
    skip_stop_words: bool
    pos_tagset: str


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

        with path.joinpath('word_vectorizer.pkl').open('rb') as f:
            word_vectorizer = pickle.load(f)

        with path.joinpath('pos_vectorizer.pkl').open('rb') as f:
            pos_vectorizer = pickle.load(f)

        with path.joinpath('config.json').open('r') as f:
            cfg_dict = json.load(f)
            cfg = CountVectorizerConfig.model_validate(cfg_dict)

        return cls(cfg, word_vectorizer, doc_frequency_vector, pos_vectorizer)

    def __init__(self,
                 cfg: CountVectorizerConfig,
                 word_vectorizer: SklearnCountVectorizer | None,
                 doc_frequency_vector: np.ndarray | None,
                 pos_vectorizer: SklearnCountVectorizer | None) -> None:

        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('universal_tagset', quiet=True)

        self._cfg = cfg

        if doc_frequency_vector is None:
            self._doc_frequency_vector = np.array([], dtype=np.int32)
        else:
            self._doc_frequency_vector = doc_frequency_vector

        if word_vectorizer is None:

            if cfg.skip_stop_words:
                stop_words = nltk.corpus.stopwords.words('english')
            else:
                stop_words = None

            self._word_vectorizer = SklearnCountVectorizer(
                max_features=cfg.max_vocabulary_size,
                stop_words=stop_words,
            )
        else:
            self._word_vectorizer = word_vectorizer

        if pos_vectorizer is None:
            self._pos_vectorizer = SklearnCountVectorizer()
        else:
            self._pos_vectorizer = pos_vectorizer

    def fit(self, docs: list[str]) -> None:
        """Fits the vectorizer to the provided documents."""

        self._word_vectorizer.fit(docs)
        self._pos_vectorizer.fit(map(self._pos_tag_document, docs))

        if self._doc_frequency_vector.size == 0:
            self._doc_frequency_vector = np.zeros(
                len(self._word_vectorizer.vocabulary_), dtype=np.int32)

        for doc in docs:
            vector = self._word_vectorizer.transform([doc])
            self._doc_frequency_vector += (vector.toarray()[0] > 0).astype(np.int32)

    def generate_word_count_vectors(self, documents: Iterable[str]) -> np.ndarray:
        """Transforms the provided documents into vectorized representations."""

        return self._word_vectorizer.transform(documents).toarray()  # type: ignore

    def generate_pos_count_vectors(self, documents: Iterable[str]) -> np.ndarray:
        """Generates POS tag count vectors for the provided documents."""

        return self._pos_vectorizer.transform(  # type: ignore
            map(self._pos_tag_document, documents)
        ).toarray()

    def serialize(self, output_dir: pathlib.Path) -> None:
        """Serializes the vectorizer to the provided path."""

        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self._doc_frequency_vector, output_dir / 'doc_frequency_vector.pt')

        with output_dir.joinpath('word_vectorizer.pkl').open('wb') as f:
            pickle.dump(self._word_vectorizer, f)

        with output_dir.joinpath('pos_vectorizer.pkl').open('wb') as f:
            pickle.dump(self._pos_vectorizer, f)

        with output_dir.joinpath('config.json').open('w') as f:
            json.dump(self._cfg.model_dump(), f)

        with output_dir.joinpath('word_count_vocabulary.json').open('w') as f:
            json.dump({w: int(cnt) for w, cnt in self._word_vectorizer.vocabulary_.items()},
                      f, indent=4, ensure_ascii=False)

        with output_dir.joinpath('pos_count_vocabulary.json').open('w') as f:
            json.dump({w: int(cnt) for w, cnt in self._pos_vectorizer.vocabulary_.items()},
                      f, indent=4, ensure_ascii=False)

    def _pos_tag_document(self, document: str) -> str:
        """Generates a space-separated string of POS tags for the given document."""

        tokens = nltk.tokenize.word_tokenize(document)
        pos_tags = nltk.pos_tag(tokens, tagset=self._cfg.pos_tagset)

        tags_only = [tag for _, tag in pos_tags]

        return ' '.join(tags_only)
