"""Google Maps reviews scraping engine."""

import logging
from typing import AsyncIterator, List

from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage

from advanced_data_mining.data.raw_ds import Restaurant
from advanced_data_mining.data.raw_ds import Review
from advanced_data_mining.data.scraping import reviews_extractor, restaurants_extractor


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class MapsBrowser:
    """Iterates over Google Maps search results and scrapes reviews from locations."""

    def __init__(self,
                 max_reviews_per_restaurant: int,
                 max_restaurants_per_location: int) -> None:

        self._max_reviews_per_restaurant = max_reviews_per_restaurant
        self._max_restaurants_per_location = max_restaurants_per_location

    def get_locations_by_query(self,
                               primary_location: str,
                               secondary_location: str,
                               page: SyncPage) -> List[Restaurant]:
        """Yields restaurants returned by Google Maps for a given location specification.

        Args:
            primary_location: E.g. "Warsaw", "Krakow"
            secondary_location: E.g. "old town"
        """

        try:
            page.goto('https://www.google.com/maps')

        except Exception as exc:  # pylint: disable=broad-except
            _logger().error('Failed to open Google Maps: %s', exc)
            return []

        extractor = restaurants_extractor.RestaurantsExtractor(
            page,
            primary_location,
            secondary_location,
            self._max_restaurants_per_location
        )

        _logger().debug('Found %d locations for query: %s',
                        extractor.n_restaurants, f"{primary_location} {secondary_location}")

        return extractor.get_restaurants()

    async def scrape_reviews_for(self,
                                 location: Restaurant,
                                 page: AsyncPage) -> AsyncIterator[Review]:
        """Yield reviews for a single location page."""

        try:
            await page.goto(location.href)

        except Exception as exc:  # pylint: disable=broad-except
            _logger().error(
                'Failed to open location page: %s, error: %s', location.href, exc
            )
            return

        extractor = await reviews_extractor.ReviewsExtractor.create(
            page,
            self._max_reviews_per_restaurant
        )

        _logger().debug('Found %d reviews for location: %s',
                        await extractor.get_n_reviews(), location.name)

        async for review in extractor.iter_reviews():
            yield review
