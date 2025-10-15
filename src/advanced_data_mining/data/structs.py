
from typing import Dict
import dataclasses


@dataclasses.dataclass
class Restaurant:
    """Represents a specific location on Google Maps."""
    href: str
    name: str
    basic_info: str

    def __hash__(self):
        return hash(self.href)


@dataclasses.dataclass
class Review:
    """Represents a review for a specific location on Google Maps."""

    text: str
    rating: float
