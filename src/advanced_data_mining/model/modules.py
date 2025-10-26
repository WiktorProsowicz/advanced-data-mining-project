"""Contains layers and modules for building models."""

from typing import List

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
                    torch.nn.LayerNorm(out_dim),
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
    """Encodes a single numerical feature."""

    def __init__(self, hidden_dims: List[int], dropout_rate: float):

        super().__init__()

        self._layers = torchkan.KAN(
            layers_hidden=[1] + hidden_dims,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.GELU
        )

        self._postnet = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dims[-1]),
            torch.nn.Dropout(dropout_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input numerical feature into dense representation."""

        return self._postnet(self._layers(x.unsqueeze(-1)))
