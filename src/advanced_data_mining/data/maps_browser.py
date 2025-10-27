# -*- coding: utf-8 -*-
"""Contains primary Google Maps reviews scraping engine."""
import logging
import re
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

from playwright.sync_api import Page
from playwright.sync_api import sync_playwright

from advanced_data_mining.data.raw_ds import Restaurant
from advanced_data_mining.data.raw_ds import Review


def _logger():
    return logging.getLogger(__name__)


class MapsBrowser:
    """Iterates over Google Maps search results and scrapes reviews from locations."""

    def __init__(self,
                 proxy_cfg: Dict[str, str],
                 max_reviews_per_restaurant: int):

        self._proxy_cfg = proxy_cfg
        self._reviews_scroll_retries = 5
        self._max_reviews_per_restaurant = max_reviews_per_restaurant

    def get_locations_by_query(self,
                               google_maps_query: str):
        """Searches Google Maps for locations matching the query and returns their URLs."""

        locations: List[Restaurant] = []

        with sync_playwright() as p:

            browser = p.firefox.launch(headless=True,
                                       proxy=self._proxy_cfg)  # type: ignore

            context = browser.new_context(
                # viewport={"width": 1280, "height": 1280},
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            )

            page = context.new_page()

            try:
                page.goto('https://www.google.com/maps', timeout=10000)

            except Exception as e:  # pylint: disable=broad-except
                _logger().error('Failed to open Google Maps: %s', e)
                return locations

            self._open_restaurants_panel(page, google_maps_query)
            self._scroll_restaurants_to_end(page)

            restaurant_divs = page.locator('div.Nv2PK.THOPZb.CpccDe')

            for i in range(restaurant_divs.count()):
                div = restaurant_divs.nth(i)

                locations.append(self._extract_location(div))

        return locations

    def scrape_reviews_for(self,
                           location: Restaurant) -> Iterator[Review]:
        """Collects all reviews for a specific location."""

        with sync_playwright() as p:

            browser = p.firefox.launch(headless=True,
                                       proxy=self._proxy_cfg)  # type: ignore

            context = browser.new_context(
                # viewport={"width": 1280, "height": 1280},
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            )

            page = context.new_page()
            page.set_default_timeout(2000)

            try:
                page.goto(location.href, timeout=10000)

            except Exception as e:  # pylint: disable=broad-except
                _logger().error('Failed to open location page: %s, error: %s',
                                location.href, e)
                return

            self._open_more_reviews(page)

            self._scroll_reviews_to_end(page)

            side_panel = page.locator('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde').first

            review_divs = side_panel.locator('div.jftiEf')

            _logger().debug('Found %d reviews for location: %s',
                            review_divs.count(), location.name)

            for i in range(review_divs.count()):

                try:
                    yield self._extract_review(review_divs.nth(i))

                except Exception as e:  # pylint: disable=broad-except
                    _logger().error('Failed to extract review: %s', e)
                    continue

    def _open_more_reviews(self, page):

        btn = page.locator('button:has-text("More reviews")')
        btn.first.click(timeout=4000)
        page.wait_for_timeout(2000)

    def _open_restaurants_panel(self, page, query):

        search_panel = page.locator('input[id="searchboxinput"]')

        if search_panel.count() > 0:
            search_panel.first.click(timeout=4000)
            page.wait_for_timeout(800)

            search_panel.first.fill(query)
            page.wait_for_timeout(800)

            search_panel.first.press('Enter')
            page.wait_for_timeout(4000)

    def _scroll_reviews_to_end(self, page):

        side_panel = page.locator('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde')

        if side_panel.count() == 0:
            _logger().critical('Cannot find reviews side panel!')
            return

        side_panel = side_panel.first

        review_divs = side_panel.locator('div.jftiEf')

        if review_divs.count() == 0:
            _logger().warning('No reviews found in the side panel!')
            return

        review_divs_count = review_divs.count()
        num_retries = self._reviews_scroll_retries

        while review_divs_count > 0:

            if review_divs_count >= self._max_reviews_per_restaurant:
                return

            page.evaluate('(el) => el.scrollTop = el.scrollHeight',
                          side_panel.element_handle())
            page.wait_for_timeout(1000)

            review_divs = side_panel.locator('div.jftiEf')
            new_review_divs_count = review_divs.count()

            _logger().debug('Scrolled reviews panel, found %d reviews so far.',
                            new_review_divs_count)

            if new_review_divs_count == review_divs_count:

                if num_retries > 0:
                    _logger().debug('No new reviews found, retrying scroll (%d retries left).',
                                    num_retries)
                    num_retries -= 1
                    continue

                _logger().debug('Scrolled to the end of reviews. Found %d reviews.',
                                new_review_divs_count)
                break

            num_retries = self._reviews_scroll_retries
            review_divs_count = new_review_divs_count

    def _scroll_restaurants_to_end(self, page):

        results_side = page.locator('div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde.ecceSd')

        if results_side.count() == 0:
            return

        results_side = results_side.nth(1)

        containers = page.locator('div.Nv2PK.THOPZb.CpccDe')

        if containers.count() == 0:
            return

        containers_count = containers.count()

        while containers_count > 0:
            page.evaluate('el => el.scrollBy(0, el.scrollHeight)',
                          results_side.element_handle())
            page.wait_for_timeout(1000)

            containers = page.locator('div.Nv2PK.THOPZb.CpccDe')
            new_containers_count = containers.count()

            if new_containers_count == containers_count:
                break

            containers_count = new_containers_count

    def _extract_location(self, restaurant_div) -> Restaurant:

        href = restaurant_div.locator('a.hfpxzc').first.get_attribute('href')

        basic_info_div = restaurant_div.locator('div.UaQhfb').first
        restaurant_name = basic_info_div.locator('div.NrDZNb').first.inner_text()

        el = basic_info_div.locator('div.W4Efsd').nth(1)
        basic_info_div = el.locator('div.W4Efsd').first
        basic_info = basic_info_div.inner_text()

        return Restaurant(href=href, name=restaurant_name, basic_info=basic_info)

    def _extract_review(self, review_div) -> Review:

        more_btn = review_div.locator('button.w8nwRe.kyuRq')
        if more_btn.count() > 0:
            more_btn.first.click(timeout=800)
            review_div.page.wait_for_timeout(120)

        stars_span = review_div.locator('span.kvMYJc').first
        rating = self._extract_rating(stars_span)

        is_translated, translated_text, original_text = self._extract_texts(review_div)

        return Review(
            text=translated_text,
            rating=rating,
            translated=is_translated,
            original=original_text
        )

    def _extract_texts(self, review_div) -> Tuple[bool, str, str]:

        """Extracts translated and original texts from the review div."""

        text_spans = review_div.locator('span.wiI7pd')
        span_texts = [span.strip() for span in text_spans.all_inner_texts() if span and span.strip()]

        translated_marker = review_div.locator('span:has-text("Translated by Google")')
        is_translated = translated_marker.count() > 0

        candidate_buttons = review_div.locator('button')
        button_texts = [text.strip() for text in candidate_buttons.all_inner_texts()
                        if text and text.strip()]

        marker_selectors = [
            'span:has-text("See original")',
            'button:has-text("See original")',
            '[aria-label*="See original"]',
            'span:has-text("Show original")',
            'button:has-text("Show original")',
            '[aria-label*="Show original"]',
        ]

        if not is_translated:
            for selector in marker_selectors:
                locator_matches = review_div.locator(selector)
                if locator_matches.count() > 0:
                    is_translated = True
                    break

        if not is_translated:
            is_translated = any('original' in text.lower() for text in button_texts)

        original_text = ''
        translated_text = span_texts[0] if span_texts else ''

        if is_translated and len(span_texts) > 1:
            translated_text = span_texts[0]
            original_text = span_texts[1]
        elif is_translated:
            """ Attempt to reveal original text by clicking the "See original" button """
            show_original_btn = review_div.locator('button:has-text("See original")')
            if show_original_btn.count() == 0:
                show_original_btn = review_div.locator('button:has-text("Show original")')

            if show_original_btn.count() > 0:
                try:
                    show_original_btn.first.click(timeout=1000)
                    review_div.page.wait_for_timeout(300)
                    refreshed_spans = [span.strip()
                    for span in review_div.locator('span.wiI7pd').all_inner_texts() if span and span.strip()]
                    if refreshed_spans:
                        original_text = refreshed_spans[-1]
                except Exception as exc: 
                    _logger().debug('Failed to click "See original": %s', exc)

        if is_translated and not original_text:
            dataset = review_div.evaluate('el => el.dataset')
            if isinstance(dataset, dict):
                original_text = dataset.get('originalReviewText', original_text)

        if not is_translated:
            original_text = ''
        elif not original_text:
            original_text = translated_text

        return is_translated, translated_text, original_text

    def _extract_rating(self, stars_span) -> float:

        aria = stars_span.get_attribute('aria-label')

        isolated_num = re.search(r'[\d]+', aria)

        if isolated_num:
            return float(isolated_num.group(0))

        return 0.0
