"""Contains layers and modules for building models."""
from typing import List
from typing import Tuple

import torch
import pydantic

from advanced_data_mining.model import torchkan


class _SequencePreNet(torch.nn.Module):
    """Pre-processes a sequence of linguistic features into a single embedding."""

    def __init__(self, input_dim: int, dropout_rate: float):

        super().__init__()

        self._att_query = torch.nn.Parameter(torch.zeros(input_dim),
                                             requires_grad=True)
        torch.nn.init.xavier_uniform_(self._att_query.unsqueeze(0))

        self._attention = torch.nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=dropout_rate,
            batch_first=True
        )

    def forward(self, input_sequence: torch.Tensor, sequence_length: torch.Tensor) -> torch.Tensor:
        """Pre-processes a sequence of linguistic features into a single embedding.

        Args:
            input_sequence: Input sequence of vectors of shape [B, T, D].
            sequence_length: Lengths of the input sequences of shape [B].
        """

        indices_sequences = (torch.arange(input_sequence.size(1), device=input_sequence.device)
                             .expand(len(sequence_length), -1))
        key_padding_mask = indices_sequences >= sequence_length.unsqueeze(1)

        att_query = self._att_query.expand(input_sequence.size(0), -1).unsqueeze(1)
        att_output, _ = self._attention(att_query,
                                        input_sequence,
                                        input_sequence,
                                        key_padding_mask)
        return att_output.squeeze(1)  # type: ignore[no-any-return]


class LinguisticEncoder(torch.nn.Module):
    """Encodes input linguistic representation (possibly sequential) into a dense vector."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for LinguisticEncoder."""
        input_dim: int
        hidden_dims: List[int]
        dropout_rate: float
        accepts_sequence: bool

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._prenet: _SequencePreNet | None = None

        if cfg.accepts_sequence:
            self._prenet = _SequencePreNet(cfg.input_dim, cfg.dropout_rate)

        self._blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.BatchNorm1d(out_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(cfg.dropout_rate)
                ) for in_dim, out_dim in zip([cfg.input_dim] + cfg.hidden_dims[:-1],
                                             cfg.hidden_dims)
            ]
        )

    def forward(self,
                x: torch.Tensor,
                sequence_length: torch.Tensor | None = None) -> torch.Tensor:
        """Encodes input linguistic representation into dense representation."""

        if self._prenet is not None:
            x = self._prenet(x, sequence_length)

        for layer in self._blocks:
            x = layer(x)
        return x


class CatFeatureEncoder(torch.nn.Module):
    """Encodes categorical features into a dense vector."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for CatFeatureEncoder."""
        input_dim: int
        hidden_dim: int

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._embedding = torch.nn.Embedding(cfg.input_dim, cfg.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input categorical features into dense representation."""

        return self._embedding(x)  # type: ignore[no-any-return]


class NumFeaturesEncoder(torch.nn.Module):
    """Encodes a single numerical feature vector."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for NumFeaturesEncoder."""
        input_dim: int
        hidden_dims: List[int]
        dropout_rate: float

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._layers = torchkan.KAN(  # type: ignore
            layers_hidden=[cfg.input_dim] + cfg.hidden_dims,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.GELU
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.BatchNorm1d(cfg.hidden_dims[-1]),
            torch.nn.Dropout(cfg.dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input numerical feature into dense representation."""

        return self._postnet(self._layers(x))  # type: ignore[no-any-return]


class PostNet(torch.nn.Module):
    """Post-processing network for combining encoded features."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for PostNet."""
        input_dim: int
        hidden_dims: List[int]
        dropout_rate: float

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.BatchNorm1d(out_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(cfg.dropout_rate)
                ) for in_dim, out_dim in zip(
                    [cfg.input_dim] + cfg.hidden_dims[:-1], cfg.hidden_dims
                )
            ]
        )

        self._classification_output = torch.nn.Linear(cfg.hidden_dims[-1], 5)
        self._translation_classification_output = torch.nn.Linear(cfg.hidden_dims[-1], 1)
        self._regression_output = torch.nn.Linear(cfg.hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Processes combined feature representation to produce final output."""

        for layer in self._layers:
            x = layer(x)

        return (self._classification_output(x),
                self._translation_classification_output(x).squeeze(1),
                self._regression_output(x).squeeze(1))
