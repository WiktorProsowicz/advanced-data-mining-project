"""Google Maps reviews scraping engine."""

import logging
from typing import Iterator

from playwright.sync_api import sync_playwright

from advanced_data_mining.data.raw_ds import Restaurant
from advanced_data_mining.data.raw_ds import Review
from advanced_data_mining.data.scraping import reviews_extractor, restaurants_extractor


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class MapsBrowser:
    """Iterates over Google Maps search results and scrapes reviews from locations."""

    def __init__(self,
                 proxy_cfg: dict[str, str],
                 max_reviews_per_restaurant: int,
                 max_restaurants_per_location: int) -> None:
        self._proxy_cfg = proxy_cfg
        self._max_reviews_per_restaurant = max_reviews_per_restaurant
        self._max_restaurants_per_location = max_restaurants_per_location

    def get_locations_by_query(self,
                               primary_location: str,
                               secondary_location: str) -> Iterator[Restaurant]:
        """Yields restaurants returned by Google Maps for a given location specification.

        Args:
            primary_location: E.g. "Warsaw", "Krakow"
            secondary_location: E.g. "old town"
        """

        with sync_playwright() as playwright:
            browser = playwright.firefox.launch(
                headless=True,
                proxy=self._proxy_cfg,  # type: ignore[arg-type]
            )
            page = browser.new_context(
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            ).new_page()

            try:
                page.goto('https://www.google.com/maps', timeout=10000)

            except Exception as exc:  # pylint: disable=broad-except
                _logger().error('Failed to open Google Maps: %s', exc)
                return

            extractor = restaurants_extractor.RestaurantsExtractor(
                page,
                primary_location,
                secondary_location,
                self._max_restaurants_per_location
            )

            _logger().debug('Found %d locations for query: %s',
                            extractor.n_restaurants, f"{primary_location} {secondary_location}")

            yield from extractor.iter_restaurants()

    def scrape_reviews_for(self, location: Restaurant) -> Iterator[Review]:
        """Yield reviews for a single location page."""
        with sync_playwright() as playwright:
            browser = playwright.firefox.launch(
                headless=True,
                proxy=self._proxy_cfg,  # type: ignore[arg-type]
            )
            page = browser.new_context(
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            ).new_page()
            page.set_default_timeout(2000)

            try:
                page.goto(location.href, timeout=10000)

            except Exception as exc:  # pylint: disable=broad-except
                _logger().error(
                    'Failed to open location page: %s, error: %s', location.href, exc
                )
                return

            extractor = reviews_extractor.ReviewsExtractor(
                page, max_reviews=self._max_reviews_per_restaurant
            )

            _logger().debug('Found %d reviews for location: %s', extractor.n_reviews, location.name)

            yield from extractor.iter_reviews()
