# -*- coding: utf-8 -*-
"""Contains definitions of raw dataset structures and utilities for loading/saving them."""
import dataclasses
import json
import os
from typing import Dict
from typing import List
from typing import TypeAlias


@dataclasses.dataclass
class Restaurant:
    """Represents a specific location on Google Maps."""
    href: str
    name: str
    basic_info: str
    city: str = ''

    def __hash__(self):
        return hash(self.href)


@dataclasses.dataclass
class Review:
    """Represents a review for a specific location on Google Maps."""

    text: str
    rating: float


RawDataset: TypeAlias = Dict[Restaurant, List[Review]]


class RawDSLoader:
    """Loads the raw dataset from the specified directory."""

    def __init__(self, raw_ds_path: str):
        self._raw_ds_path = raw_ds_path

    def load_dataset(self) -> RawDataset:
        """Loads the raw dataset from JSON files in the specified directory."""

        ds: RawDataset = {}

        for json_file in os.listdir(self._raw_ds_path):

            if any(city in json_file for city in ['krakow', 'kraków']):
                city = 'Krakow'

            else:
                city = 'Warsaw'

            with open(os.path.join(self._raw_ds_path, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)

            location = Restaurant(
                href=data['location']['href'],
                name=data['location']['name'],
                basic_info=data['location']['basic_info'],
                city=city
            )

            reviews = [
                Review(
                    text=review['text'],
                    rating=review['rating']
                ) for review in data['reviews']
            ]

            ds[location] = reviews

        return ds
