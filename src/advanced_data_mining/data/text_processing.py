"""Contains utilities for text processing, normalization and obtaining embeddings."""

from typing import List, Set, DefaultDict, Tuple
import collections

import torch
import nltk  # type: ignore
import gruut
import tqdm  # type: ignore
import numpy as np


class TextPreprocessor:
    """Preprocesses text data for further analysis."""

    def __init__(self):

        self._vocabulary: DefaultDict[str, int] = collections.defaultdict(int)  # type: ignore
        self._sorted_vocabulary: List[str] = []  # type: ignore
        self._pos_vocab: Set[str] = set()

        nltk.download('punkt_tab', quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        self._stop_words = set(nltk.corpus.stopwords.words("english"))

    @property
    def vocabulary(self) -> DefaultDict[str, int]:
        """Returns the internal vocabulary with word counts."""
        return self._vocabulary

    @staticmethod
    def save_vocabulary_to_file(vocabulary: DefaultDict[str, int], filepath: str):
        """Saves the vocabulary to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for word, count in sorted(vocabulary.items(),
                                      key=lambda item: item[1],
                                      reverse=True):
                f.write(f"{word};{count}\n")

    def num_words(self, text: str) -> int:
        """Returns the number of words in the given text."""
        words = nltk.tokenize.word_tokenize(text)
        return len(words)

    def num_sentences(self, text: str) -> int:
        """Returns the number of sentences in the given text."""
        sentences = nltk.tokenize.sent_tokenize(text)
        return len(sentences)

    def normalize_text(self, text: str) -> str:
        """Normalizes text.

        Normalization includes converting currency, numbers to words, and standardizing punctuation.
        """

        sentences = [sentence.text_with_ws for sentence in gruut.sentences(text, phonemes=False)]
        return " ".join(sentences)

    def update_vocabulary(self, texts: List[str]):
        """Updates the internal vocabulary with words from the provided texts."""

        for text in tqdm.tqdm(texts, desc="Building vocabulary", total=len(texts)):
            words = self._prepare_for_bow(text)
            for word in words:
                self._vocabulary[word] += 1

        self._sorted_vocabulary = sorted(
            self._vocabulary.keys(),
            key=lambda word: self._vocabulary[word],
            reverse=True
        )

    def get_pos_bow_representation(self, text: str) -> torch.Tensor:
        """Generates a bag-of-words representation based on parts of speech for the given text."""

        words = self._prepare_for_bow(text)
        pos_tags = nltk.pos_tag(words)

        pos_to_index = {pos: idx for idx, pos in enumerate(sorted(self._pos_vocab))}

        bow_vector = np.zeros(len(self._pos_vocab), dtype=np.int32)

        for _, pos in pos_tags:
            if pos in self._pos_vocab:
                index = pos_to_index[pos]
                bow_vector[index] += 1

        return torch.tensor(bow_vector)

    def update_pos_vocab(self, texts: List[str]):
        """Updates the internal vocabulary with parts of speech from the provided texts."""

        for text in tqdm.tqdm(texts, desc="Building POS vocabulary", total=len(texts)):

            words = self._prepare_for_bow(text)
            pos_tags = nltk.pos_tag(words)

            for _, pos in pos_tags:
                self._pos_vocab.add(pos)

    def save_pos_vocab_to_file(self, filepath: str):
        """Saves the POS vocabulary to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for pos in sorted(self._pos_vocab):
                f.write(f"{pos}\n")

    def load_pos_vocab_from_file(self, filepath: str):
        """Loads POS vocabulary from a file."""
        self._pos_vocab.clear()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                pos = line.strip()
                self._pos_vocab.add(pos)

    def top_bottom_n_words(self, n: int) -> Tuple[DefaultDict[str, int], ...]:
        """Returns the top/bottom N words in the vocabulary based on their frequency."""

        return (
            self._take_n_words_from_vocab(n, from_top=True),
            self._take_n_words_from_vocab(n, from_top=False)
        )

    def get_bow_representation(self, text: str) -> torch.Tensor:
        """Generates a bag-of-words representation for the given text."""

        words = set(self._prepare_for_bow(text))

        bow_vector = np.zeros(len(self._vocabulary), dtype=np.float32)

        for word in words:
            if word in self._sorted_vocabulary:
                index = self._sorted_vocabulary.index(word)
                bow_vector[index] = 1.0

        return torch.tensor(bow_vector).to_sparse()

    def load_vocab_from_file(self, filepath: str):
        """Loads vocabulary from a file."""

        self._vocabulary.clear()
        self._sorted_vocabulary.clear()

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                word, count = line.strip().split(";")
                self._vocabulary[word] = int(count)

        self._sorted_vocabulary = sorted(
            self._vocabulary.keys(),
            key=lambda word: self._vocabulary[word],
            reverse=True
        )

    def _take_n_words_from_vocab(self, n: int, from_top: bool = True) -> DefaultDict[str, int]:
        """Keeps only the top/bottom N words in the vocabulary."""

        if from_top:
            words_to_keep = set(self._sorted_vocabulary[:n])
        else:
            words_to_keep = set(self._sorted_vocabulary[-n:])

        return collections.defaultdict(
            int,
            {word: count for word, count in self._vocabulary.items() if word in words_to_keep}
        )

    def _prepare_for_bow(self, text: str) -> List[str]:
        """Prepares text for bag-of-words representation."""

        words = nltk.tokenize.word_tokenize(text.lower())
        words = [word for word in words if word not in self._stop_words]
        words = [word for word in words if word.isalpha()]
        return words
