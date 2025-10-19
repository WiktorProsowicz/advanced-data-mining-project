"""Contains utilities for text processing, normalization and obtaining embeddings."""

from typing import List, Set, DefaultDict, Tuple, Optional
import collections
import dataclasses

import torch
import nltk  # type: ignore
import gruut
import tqdm  # type: ignore
import numpy as np
from transformers import BertTokenizer, BertModel  # type: ignore


@dataclasses.dataclass
class Vocabulary:
    """Holds vocabulary composed from a corpus of texts."""

    word_counts: DefaultDict[str, int]
    sorted_words: List[str]
    word_in_doc_counts: DefaultDict[str, int]
    n_docs: int


class TextPreprocessor:
    """Preprocesses text data for further analysis."""

    def __init__(self, bert_model_device: str = "cpu"):

        self._vocabulary = Vocabulary(
            word_counts=collections.defaultdict(int),
            sorted_words=[],
            word_in_doc_counts=collections.defaultdict(int),
            n_docs=0
        )

        self._pos_vocab: Set[str] = set()

        self._bert_model_device = bert_model_device

        nltk.download('punkt_tab', quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        self._stop_words = set(nltk.corpus.stopwords.words("english"))

        self._bert_tokenizer: Optional[BertTokenizer] = None
        self._bert_model: Optional[BertModel] = None

    @property
    def vocabulary(self) -> Vocabulary:
        """Returns the internal vocabulary with word counts."""
        return self._vocabulary

    @staticmethod
    def save_vocabulary_to_file(vocabulary: Vocabulary, filepath: str):
        """Saves the vocabulary to a file."""
        with open(filepath, "w", encoding="utf-8") as f:

            f.write(str(vocabulary.n_docs) + "\n")

            for word, count in sorted(vocabulary.word_counts.items(),
                                      key=lambda item: item[1],
                                      reverse=True):

                f.write(f"{word};{count};{vocabulary.word_in_doc_counts[word]}\n")

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

    def get_bert_embeddings(self, text: str) -> List[torch.Tensor]:
        """Generates word-level BERT embeddings for the given text's sentences."""

        embeddings: List[torch.Tensor] = []

        tokenizer, model = self._get_bert_model_and_tokenizer()

        for sentence in nltk.tokenize.sent_tokenize(text):

            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state.squeeze(0)[1:-1])

        return embeddings

    def calc_trace_velocity(self,
                            sentence_embeddings: List[torch.Tensor],
                            chunk_length: int) -> float:
        """Calculates trace velocity for the given sentence embeddings."""

        cat_embeddings = torch.cat(sentence_embeddings, dim=0)

        chunks = [cat_embeddings[i:i + chunk_length]
                  for i in range(0, len(cat_embeddings), chunk_length)
                  if i + chunk_length <= len(cat_embeddings)]

        if len(chunks) < 2:
            return 0.0

        centroids = [chunk.mean(dim=0) for chunk in chunks]

        velocities = [torch.norm(centroids[i] - centroids[i - 1]).item()
                      for i in range(1, len(centroids))]

        return np.mean(velocities).item()

    def calc_trace_volume(self,
                          sentence_embeddings: List[torch.Tensor],
                          chunk_length: int) -> float:
        """Calculates trace volume for the given sentence embeddings."""

        cat_embeddings = torch.cat(sentence_embeddings, dim=0)

        chunks = [cat_embeddings[i:i + chunk_length]
                  for i in range(0, len(cat_embeddings), chunk_length)
                  if i + chunk_length <= len(cat_embeddings)]

        if len(chunks) < 2:
            return 0.0

        min_values = torch.min(cat_embeddings, dim=0).values
        max_values = torch.max(cat_embeddings, dim=0).values

        return torch.norm(max_values - min_values).item()

    def _get_bert_model_and_tokenizer(self) -> Tuple[BertTokenizer, BertModel]:
        """Initializes and returns BERT tokenizer and model."""

        device = torch.device(self._bert_model_device)

        if self._bert_tokenizer is None or self._bert_model is None:
            self._bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self._bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

        return self._bert_tokenizer, self._bert_model

    def update_vocabulary(self, texts: List[str]):
        """Updates the internal vocabulary with words from the provided texts."""

        for text in tqdm.tqdm(texts, desc="Building vocabulary", total=len(texts)):
            words = self._prepare_for_bow(text)

            for word in words:
                self._vocabulary.word_counts[word] += 1

            for word in set(words):
                self._vocabulary.word_in_doc_counts[word] += 1

            self._vocabulary.n_docs += 1

        self._vocabulary.sorted_words = sorted(
            self._vocabulary.word_counts.keys(),
            key=lambda word: self._vocabulary.word_counts[word],
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

    def top_bottom_n_words(self, n: int) -> Tuple[Vocabulary, Vocabulary]:
        """Returns the top/bottom N words in the vocabulary based on their frequency."""

        return (
            self._take_n_words_from_vocab(n, from_top=True),
            self._take_n_words_from_vocab(n, from_top=False)
        )

    def get_bow_representation(self, text: str) -> torch.Tensor:
        """Generates a bag-of-words representation for the given text."""

        words = set(self._prepare_for_bow(text))

        bow_vector = np.zeros(len(self._vocabulary.word_counts), dtype=np.float32)

        for word in words:
            if word in self._vocabulary.word_counts:
                index = self._vocabulary.sorted_words.index(word)
                bow_vector[index] = 1.0

        return torch.tensor(bow_vector).to_sparse()

    def get_tfidf_representation(self, text: str) -> torch.Tensor:
        """Generates a TF-IDF representation for the given text."""

        words = self._prepare_for_bow(text)
        words = [word for word in words if word in self._vocabulary.word_counts]
        word_count = collections.Counter(words)
        total_words = len(words)

        tfidf_vector = np.zeros(len(self._vocabulary.word_counts), dtype=np.float32)

        for word, count in word_count.items():
            if word in self._vocabulary.sorted_words:
                index = self._vocabulary.sorted_words.index(word)

                tf = count / total_words
                idf = np.log(1 + self._vocabulary.n_docs /
                             (1 + self._vocabulary.word_in_doc_counts[word]))

                tfidf_vector[index] = tf * idf

        return torch.tensor(tfidf_vector).to_sparse()

    def load_vocab_from_file(self, filepath: str):
        """Loads vocabulary from a file."""

        self._vocabulary.word_counts.clear()
        self._vocabulary.word_in_doc_counts.clear()
        self._vocabulary.sorted_words.clear()
        self._vocabulary.n_docs = 0

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

            self._vocabulary.n_docs = int(lines[0].strip())

            for line in lines[1:]:
                word, count, doc_count = line.strip().split(";")
                self._vocabulary.word_counts[word] = int(count)
                self._vocabulary.word_in_doc_counts[word] = int(doc_count)

        self._vocabulary.sorted_words = sorted(
            self._vocabulary.word_counts.keys(),
            key=lambda word: self._vocabulary.word_counts[word],
            reverse=True
        )

    def _take_n_words_from_vocab(self,
                                 n: int,
                                 from_top: bool = True) -> Vocabulary:
        """Keeps only the top/bottom N words in the vocabulary."""

        if from_top:
            words_to_keep = set(self._vocabulary.sorted_words[:n])
        else:
            words_to_keep = set(self._vocabulary.sorted_words[-n:])

        return Vocabulary(
            word_counts=collections.defaultdict(
                int,
                {word: count
                 for word, count in self._vocabulary.word_counts.items()
                 if word in words_to_keep}),
            sorted_words=[word for word in self._vocabulary.sorted_words
                          if word in words_to_keep],
            word_in_doc_counts=collections.defaultdict(
                int,
                {word: count
                 for word, count in self._vocabulary.word_in_doc_counts.items()
                 if word in words_to_keep}),
            n_docs=self._vocabulary.n_docs
        )

    def _prepare_for_bow(self, text: str) -> List[str]:
        """Prepares text for bag-of-words representation."""

        words = nltk.tokenize.word_tokenize(text.lower())
        words = [word for word in words if word not in self._stop_words]
        words = [word for word in words if word.isalpha()]
        return words
