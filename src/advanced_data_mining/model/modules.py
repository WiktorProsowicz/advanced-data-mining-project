"""Contains layers and modules for building models."""
from typing import List

import torch
import pydantic

from advanced_data_mining.model import torchkan


class SequencePreNet(torch.nn.Module):
    """Pre-processes a sequence of linguistic features into a single embedding."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for SequencePreNet."""
        input_dim: int
        dropout_rate: float
        num_attention_heads: int
        num_self_attention_blocks: int
        lstm_hidden_dim: int

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._att_blocks = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(
                embed_dim=cfg.input_dim,
                num_heads=cfg.num_attention_heads,
                dropout=cfg.dropout_rate,
                batch_first=True
            ) for _ in range(cfg.num_self_attention_blocks)
        ])

        self._att_block_norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(cfg.input_dim) for _ in range(cfg.num_self_attention_blocks)
        ])

        self._att_dropout = torch.nn.Dropout(cfg.dropout_rate)

        self._sequence_squeezer = torch.nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=cfg.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self._output_projection = torch.nn.Linear(2 * cfg.lstm_hidden_dim, cfg.input_dim)

    def forward(self, input_sequence: torch.Tensor, sequence_length: torch.Tensor) -> torch.Tensor:
        """Pre-processes a sequence of linguistic features into a single embedding.

        Args:
            input_sequence: Input sequence of vectors of shape [B, T, D].
            sequence_length: Lengths of the input sequences of shape [B].
        """

        indices_sequences = (torch.arange(input_sequence.size(1), device=input_sequence.device)
                             .expand(len(sequence_length), -1))
        key_padding_mask = indices_sequences >= sequence_length.unsqueeze(1)

        x = input_sequence
        for att_block, norm in zip(self._att_blocks, self._att_block_norms):
            att_output, _ = att_block(x,
                                      x,
                                      x,
                                      key_padding_mask=key_padding_mask,
                                      need_weights=False)
            x = norm(x + self._att_dropout(att_output))

        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=sequence_length.to(torch.long).detach().cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (hidden_states, _) = self._sequence_squeezer(packed_sequence)

        forward_state = hidden_states[-2]
        backward_state = hidden_states[-1]
        squeezed = torch.cat((forward_state, backward_state), dim=-1)
        return self._output_projection(squeezed)  # type: ignore[no-any-return]


class LinguisticEncoder(torch.nn.Module):
    """Encodes input linguistic representation (possibly sequential) into a dense vector."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for LinguisticEncoder."""
        input_dim: int
        hidden_dims: List[int]
        dropout_rate: float
        sequence_prenet_cfg: SequencePreNet.Configuration | None = None

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._prenet: SequencePreNet | None = None

        if cfg.sequence_prenet_cfg is not None:
            self._prenet = SequencePreNet(cfg.sequence_prenet_cfg)

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
            if sequence_length is None:
                raise ValueError('`sequence_length` must be provided for sequential input.')
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
        kan_grid_size: int
        kan_spline_order: int

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._layers = torchkan.KAN(  # type: ignore
            layers_hidden=[cfg.input_dim] + cfg.hidden_dims,
            grid_size=cfg.kan_grid_size,
            spline_order=cfg.kan_spline_order,
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
        kan_grid_size: int
        kan_spline_order: int

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._layers = torchkan.KAN(  # type: ignore
            layers_hidden=[cfg.input_dim] + cfg.hidden_dims,
            grid_size=cfg.kan_grid_size,
            spline_order=cfg.kan_spline_order,
            base_activation=torch.nn.GELU
        )

        self._classification_output = torch.nn.Linear(cfg.hidden_dims[-1],
                                                      5, bias=False)
        self._translation_classification_output = torch.nn.Linear(cfg.hidden_dims[-1],
                                                                  1, bias=False)
        self._regression_output = torch.nn.Linear(cfg.hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Processes combined feature representation to produce final output."""

        x = self._layers(x)

        return (self._classification_output(x),
                self._translation_classification_output(x).squeeze(1),
                self._regression_output(x).squeeze(1))
