"""Module that provides utilities for generating and handling data embeddings."""

import dataclasses
import logging
from typing import Iterator

import pydantic
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import nltk


class EmbeddingGeneratorConfig(pydantic.BaseModel):
    """Configuration for the EmbeddingGenerator."""
    model_name: str
    max_sequence_length: int


@dataclasses.dataclass
class Embeddings:
    """Holds word-level and sentence-level embeddings."""
    word_embeddings: list[torch.Tensor]
    sentence_embeddings: list[torch.Tensor]


@dataclasses.dataclass
class _DocumentBatch:
    """Holds a batch of documents for processing."""
    documents: list[str]
    bert_inputs: dict[str, torch.Tensor]
    doc_indices: list[int]
    doc_lengths: list[int]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates dense BERT embeddings for given text data."""

    def __init__(self,
                 cfg: EmbeddingGeneratorConfig,
                 batch_size: int,
                 device: str):

        self._cfg = cfg
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)  # type: ignore
        self._model = AutoModel.from_pretrained(cfg.model_name).to(device)
        self._batch_size = batch_size

    @property
    def cfg(self) -> EmbeddingGeneratorConfig:
        """Returns the configuration of the EmbeddingGenerator."""
        return self._cfg

    def get_bert_embeddings(self, documents: list[str]) -> list[Embeddings]:
        """Generates word-level and sentence-level embeddings for the given documents."""

        try:
            return self._generate_embeddings_for_docs(documents, self._batch_size)

        except torch.OutOfMemoryError:
            _logger().warning('Cought OOM error while processing documents in batches.')
            return self._generate_embeddings_for_docs(documents, batch_size=1)

    def _generate_embeddings_for_docs(self,
                                      documents: list[str],
                                      batch_size: int) -> list[Embeddings]:

        embeddings_list: list[Embeddings] = []

        for doc_batch in self._batch_documents(documents, batch_size):

            with torch.no_grad():
                outputs = self._model(**doc_batch.bert_inputs)

            attention_mask = doc_batch.bert_inputs['attention_mask']

            for doc_idx, doc_length in zip(doc_batch.doc_indices, doc_batch.doc_lengths):

                embeddings = Embeddings(word_embeddings=[],
                                        sentence_embeddings=[])

                for i, sentence_hidden_states in enumerate(
                        outputs.last_hidden_state[doc_idx:doc_idx + doc_length]):

                    sentence_size = int(torch.sum(attention_mask[doc_idx + i]).item()) - 2

                    sentence_embedding = sentence_hidden_states[0]
                    token_embeddings = sentence_hidden_states[1: sentence_size + 1]

                    embeddings.word_embeddings.append(token_embeddings)
                    embeddings.sentence_embeddings.append(sentence_embedding)

                embeddings_list.append(embeddings)

        return embeddings_list

    def _batch_documents(self, documents: list[str], batch_size: int) -> Iterator[_DocumentBatch]:
        """Batches documents for processing."""

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i: i + batch_size]

            sentences: list[str] = []
            doc_indices: list[int] = []
            doc_lengths: list[int] = []

            for doc in batch_docs:
                doc_sentences = nltk.tokenize.sent_tokenize(doc)
                sentences.extend(doc_sentences)
                doc_indices.append(sum(doc_lengths))
                doc_lengths.append(len(doc_sentences))

            inputs = self._tokenizer(sentences,
                                     return_tensors='pt',
                                     truncation=True,
                                     padding=True,
                                     max_length=self._cfg.max_sequence_length)

            yield _DocumentBatch(documents=batch_docs,
                                 bert_inputs={k: v.to(self._model.device)
                                              for k, v in inputs.items()},
                                 doc_indices=doc_indices,
                                 doc_lengths=doc_lengths)
