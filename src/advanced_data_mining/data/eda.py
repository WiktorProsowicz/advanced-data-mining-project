"""Contains utilities for exploratory data analysis (EDA)."""

import json
import os
from typing import List, Dict, Any

from advanced_data_mining.data.structs import Review, Restaurant


class EDAFeatureExtractor:
    """Extracts features from the dataset for EDA purposes."""

    def __init__(self, raw_ds_path: str):

        self._raw_ds_path = raw_ds_path

    def extract_basic_stats(self) -> Dict[str, Any]:
        """Extracts basic statistics from the dataset."""

        stats: Dict[str, Any] = {}

        ds = self._load_ds()

        stats['total_restaurants'] = len(ds)
        stats['total_reviews'] = sum(len(reviews) for reviews in ds.values())
        stats['avg_reviews_per_restaurant'] = stats['total_reviews'] / stats['total_restaurants']

        restaurant_ratings = [
            sum(review.rating for review in reviews) / len(reviews) for reviews in ds.values()
        ]

        stats['avg_restaurant_rating'] = sum(restaurant_ratings) / len(restaurant_ratings)
        stats['min_restaurant_rating'] = min(restaurant_ratings)
        stats['max_restaurant_rating'] = max(restaurant_ratings)

        return stats

    def _load_ds(self) -> Dict[Restaurant, List[Review]]:

        ds: Dict[Restaurant, List[Review]] = {}

        for json_file in os.listdir(self._raw_ds_path):

            with open(os.path.join(self._raw_ds_path, json_file), "r", encoding="utf-8") as f:
                data = json.load(f)

            location = Restaurant(
                href=data["location"]["href"],
                name=data["location"]["name"],
                basic_info=data["location"]["basic_info"]
            )

            reviews = [
                Review(
                    text=review["text"],
                    rating=review["rating"]
                ) for review in data["reviews"]
            ]

            ds[location] = reviews

        return ds
