"""Contains definitions of raw dataset structures and utilities for loading/saving them."""

import json
import os
from typing import Dict, Optional
from typing import List
from typing import TypeAlias

import pydantic


class Restaurant(pydantic.BaseModel):
    """Represents a specific location on Google Maps."""
    href: str
    name: str
    basic_info: str
    # E.g. "Warsaw", "Krakow"
    primary_location: str
    # E.g. "old town"
    secondary_location: str

    def __hash__(self) -> int:
        return hash(self.href)


class Author(pydantic.BaseModel):
    """Represents the author of a review on Google Maps."""
    name: str
    n_reviews: Optional[int]


class Review(pydantic.BaseModel):
    """Represents a review for a specific restaurant on Google Maps."""
    text: str
    original: Optional[str]
    rating: float
    author: Author


RawDataset: TypeAlias = Dict[Restaurant, List[Review]]


class RawDSLoader:
    """Loads the raw dataset from the specified directory."""

    def __init__(self, raw_ds_path: str):
        self._raw_ds_path = raw_ds_path

    def load_dataset(self) -> RawDataset:
        """Loads the raw dataset from JSON files in the specified directory."""

        ds: RawDataset = {}

        for json_file in os.listdir(self._raw_ds_path):

            with open(os.path.join(self._raw_ds_path, json_file), encoding='utf-8') as f:
                data = json.load(f)

            location = Restaurant(**data['location'])

            ds[location] = [Review(**review) for review in data['reviews']]

        return ds
