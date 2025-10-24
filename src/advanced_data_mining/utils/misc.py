"""Contains miscellaneous utility functions."""

import hashlib


def hash_restaurant_href(restaurant_href: str) -> str:
    """Generates a hash for a restaurant href."""

    return hashlib.sha256(bytes(restaurant_href, encoding='utf-8')).hexdigest()
