"""Module that extracts restaurants from Google Maps page corresponding with a given query."""
import logging
from typing import Iterator
from typing import List

from playwright.sync_api import Locator
from playwright.sync_api import Page

from advanced_data_mining.data.raw_ds import Restaurant


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


_RESTAURANT_CARD_SELECTOR = 'div.Nv2PK.THOPZb.CpccDe'


class RestaurantsExtractor:
    """Extracts restaurants from Google Maps search results page."""

    def __init__(self,
                 page: Page,
                 primary_location: str,
                 secondary_location: str,
                 max_restaurants: int) -> None:

        self._primary_location = primary_location
        self._secondary_location = secondary_location
        self._max_restaurants = max_restaurants

        self._open_restaurants_panel(page, f'{primary_location} {secondary_location} kebab')
        self._scroll_restaurants_to_end(page)

        self._restaurant_divs = page.locator(_RESTAURANT_CARD_SELECTOR)

    @property
    def n_restaurants(self) -> int:
        """Returns the number of restaurant cards found on the page."""
        return self._restaurant_divs.count()

    def get_restaurants(self) -> List[Restaurant]:
        """Returns the restaurants extracted from the page."""

        n_restaurants_to_take = min(self._restaurant_divs.count(), self._max_restaurants)

        restaurants = []
        for i in range(n_restaurants_to_take):
            restaurants.append(self._extract_location(self._restaurant_divs.nth(i)))

        return restaurants

    def _open_restaurants_panel(self, page: Page, query: str) -> None:
        search_panel = page.locator('input[id="UGojuc"]')
        if search_panel.count() == 0:
            _logger().error('Could not find search panel on Google Maps page!')

        search_panel.first.click(timeout=4000)
        page.wait_for_timeout(800)

        search_panel.first.fill(query)
        page.wait_for_timeout(800)

        search_panel.first.press('Enter')
        page.wait_for_timeout(4000)

    def _scroll_restaurants_to_end(self, page: Page) -> None:
        results_side = page.locator(
            'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde.ecceSd'
        )

        if results_side.count() == 0:
            return

        results_side = results_side.nth(1)
        containers = page.locator(_RESTAURANT_CARD_SELECTOR)

        if containers.count() == 0:
            return

        containers_count = containers.count()

        while containers_count > 0:
            page.evaluate(
                'el => el.scrollBy(0, el.scrollHeight)', results_side.element_handle()
            )
            page.wait_for_timeout(1000)

            containers = page.locator(_RESTAURANT_CARD_SELECTOR)
            new_count = containers.count()

            if new_count == containers_count:
                break

            containers_count = new_count

    def _extract_location(self, restaurant_div: Locator) -> Restaurant:
        href = restaurant_div.locator('a.hfpxzc').first.get_attribute('href')

        if href is None:
            _logger().warning('Restaurant card missing href attribute!')
            href = ''

        basic_info_div = restaurant_div.locator('div.UaQhfb').first
        restaurant_name = basic_info_div.locator('div.NrDZNb').first.inner_text()

        el = basic_info_div.locator('div.W4Efsd').nth(1)
        basic_info_div = el.locator('div.W4Efsd').first
        basic_info = basic_info_div.inner_text()

        return Restaurant(href=href,
                          name=restaurant_name,
                          basic_info=basic_info,
                          primary_location=self._primary_location,
                          secondary_location=self._secondary_location)
