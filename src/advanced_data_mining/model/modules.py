"""Contains layers and modules for building models."""
from typing import List
from typing import Tuple

import torch

from advanced_data_mining.model import torchkan


class BOWEncoder(torch.nn.Module):
    """Encodes Bag-of-Words representation into a dense vector."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout_rate: float):

        super().__init__()

        self._layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(out_dim),
                    torch.nn.Dropout(dropout_rate)
                ) for in_dim, out_dim in zip(
                    [input_dim] + hidden_dims[:-1], hidden_dims
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input BOW vector into dense representation."""

        for layer in self._layers:
            x = layer(x)
        return x


class NumFeaturesEncoder(torch.nn.Module):
    """Encodes a single numerical feature vector."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout_rate: float):

        super().__init__()

        self._layers = torchkan.KAN(  # type: ignore
            layers_hidden=[input_dim] + hidden_dims,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.GELU
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hidden_dims[-1]),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input numerical feature into dense representation."""

        return self._postnet(self._layers(x))  # type: ignore[no-any-return]


class PostNet(torch.nn.Module):
    """Post-processing network for combining encoded features."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 dropout_rate: float):

        super().__init__()

        self._layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(out_dim),
                    torch.nn.Dropout(dropout_rate)
                ) for in_dim, out_dim in zip(
                    [input_dim] + hidden_dims[:-1], hidden_dims
                )
            ]
        )

        self._regression_output = torch.nn.Linear(hidden_dims[-1], 1)
        self._classification_output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], 5),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes combined feature representation to produce final output."""

        for layer in self._layers:
            x = layer(x)

        return self._regression_output(x).squeeze(-1), self._classification_output(x)
