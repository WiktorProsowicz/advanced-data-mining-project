"""Module that provides utilities for generating and handling data embeddings."""

import pydantic
from transformers import AutoModel
from transformers import AutoTokenizer
import torch
import nltk


class EmbeddingGeneratorConfig(pydantic.BaseModel):
    """Configuration for the EmbeddingGenerator."""
    model_name: str


class EmbeddingGenerator:
    """Generates dense BERT embeddings for given text data."""

    def __init__(self,
                 cfg: EmbeddingGeneratorConfig,
                 device: str):

        self._cfg = cfg
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)  # type: ignore
        self._model = AutoModel.from_pretrained(cfg.model_name).to(device)

    @property
    def cfg(self) -> EmbeddingGeneratorConfig:
        """Returns the configuration of the EmbeddingGenerator."""
        return self._cfg

    def get_bert_embeddings(self, text: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Generates word-level and sentence-level embeddings for the given text's sentences."""

        token_embeddings_list: list[torch.Tensor] = []
        sentence_embeddings_list: list[torch.Tensor] = []

        sentences = nltk.tokenize.sent_tokenize(text)

        inputs = self._tokenizer(sentences,
                                 return_tensors='pt',
                                 truncation=True,
                                 padding=True,
                                 max_length=512)

        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        for i, sentence_hidden_states in enumerate(outputs.last_hidden_state):
            sentence_size = int(torch.sum(inputs['attention_mask'][i]).item()) - 2

            sentence_embedding = sentence_hidden_states[0]
            token_embeddings = sentence_hidden_states[1: sentence_size + 1]

            token_embeddings_list.append(token_embeddings)
            sentence_embeddings_list.append(sentence_embedding)

        return token_embeddings_list, sentence_embeddings_list
