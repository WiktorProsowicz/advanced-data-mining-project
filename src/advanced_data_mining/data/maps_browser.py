"""Google Maps reviews scraping engine."""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable
from typing import Iterator

from playwright.sync_api import Locator
from playwright.sync_api import Page
from playwright.sync_api import sync_playwright

from advanced_data_mining.data.raw_ds import Restaurant
from advanced_data_mining.data.raw_ds import Review


def _logger():
    return logging.getLogger(__name__)


_RESTAURANT_CARD_SELECTOR = 'div.Nv2PK.THOPZb.CpccDe'
_REVIEWS_CONTAINER_SELECTOR = 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde'
_REVIEW_SELECTOR = 'div.jftiEf'
_SHOW_ORIGINAL_SELECTORS = (
    'button:has-text("See original")',
    'button:has-text("Show original")',
)
_TRANSLATED_MARKER_SELECTOR = 'span:has-text("Translated by Google")'


@dataclass
class ReviewTexts:
    """Holds the textual fragments extracted from a review block."""

    is_translated: bool
    translated: str
    original: str

    def harmonize(self, normalizer: Callable[[str], str]) -> None:
        """
        Ensure the translated/original pair is internally consistent.

        If both texts normalize to the same value, treat the review as non-translated
        and blank the original field to avoid duplicating content downstream.
        """
        if not self.is_translated:
            self.original = ''
            return

        if not self.original:
            return

        if normalizer(self.translated) == normalizer(self.original):
            self.is_translated = False
            self.original = ''


class MapsBrowser:
    """Iterates over Google Maps search results and scrapes reviews from locations."""

    def __init__(self, proxy_cfg: dict[str, str], max_reviews_per_restaurant: int):
        self._proxy_cfg = proxy_cfg
        self._reviews_scroll_retries = 5
        self._max_reviews_per_restaurant = max_reviews_per_restaurant

    def get_locations_by_query(self, google_maps_query: str) -> list[Restaurant]:
        """Fetch location cards returned by Google Maps for a search query."""
        locations: list[Restaurant] = []

        with sync_playwright() as playwright:
            browser = playwright.firefox.launch(
                headless=True,
                proxy=self._proxy_cfg,  # type: ignore[arg-type]
            )
            context = browser.new_context(
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            )
            page = context.new_page()

            try:
                page.goto('https://www.google.com/maps', timeout=10000)
            except Exception as exc:  # pylint: disable=broad-except
                _logger().error('Failed to open Google Maps: %s', exc)
                return locations

            self._open_restaurants_panel(page, google_maps_query)
            self._scroll_restaurants_to_end(page)

            restaurant_divs = page.locator(_RESTAURANT_CARD_SELECTOR)

            for i in range(restaurant_divs.count()):
                locations.append(self._extract_location(restaurant_divs.nth(i)))

        return locations

    def scrape_reviews_for(self, location: Restaurant) -> Iterator[Review]:
        """Yield reviews for a single location page."""
        with sync_playwright() as playwright:
            browser = playwright.firefox.launch(
                headless=True,
                proxy=self._proxy_cfg,  # type: ignore[arg-type]
            )
            context = browser.new_context(
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            )
            page = context.new_page()
            page.set_default_timeout(2000)

            try:
                page.goto(location.href, timeout=10000)
            except Exception as exc:  # pylint: disable=broad-except
                _logger().error(
                    'Failed to open location page: %s, error: %s', location.href, exc
                )
                return

            self._open_more_reviews(page)
            self._scroll_reviews_to_end(page)

            side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR).first
            review_divs = side_panel.locator(_REVIEW_SELECTOR)

            _logger().debug(
                'Found %d reviews for location: %s',
                review_divs.count(),
                location.name,
            )

            for i in range(review_divs.count()):
                review_div = review_divs.nth(i)
                try:
                    review = self._extract_review(review_div)
                    if review is not None:
                        yield review
                except Exception as exc:  # pylint: disable=broad-except
                    _logger().error('Failed to extract review: %s', exc)

    # ----------------------------------------------------------------- Page helpers
    def _open_more_reviews(self, page: Page) -> None:
        button = page.locator('button:has-text("More reviews")')
        if button.count() == 0:
            return
        try:
            button.first.click(timeout=4000)
            page.wait_for_timeout(2000)
        except Exception as exc:  # pylint: disable=broad-except
            _logger().debug('Failed to click More reviews button: %s', exc)

    def _open_restaurants_panel(self, page: Page, query: str) -> None:
        search_panel = page.locator('input[id="searchboxinput"]')
        if search_panel.count() == 0:
            return

        search_panel.first.click(timeout=4000)
        page.wait_for_timeout(800)

        search_panel.first.fill(query)
        page.wait_for_timeout(800)

        search_panel.first.press('Enter')
        page.wait_for_timeout(4000)

    def _scroll_reviews_to_end(self, page: Page) -> None:
        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR)
        if side_panel.count() == 0:
            _logger().critical('Cannot find reviews side panel!')
            return

        side_panel = side_panel.first
        review_divs = side_panel.locator(_REVIEW_SELECTOR)

        if review_divs.count() == 0:
            _logger().warning('No reviews found in the side panel!')
            return

        review_divs_count = review_divs.count()
        retries_left = self._reviews_scroll_retries

        while review_divs_count > 0:
            if review_divs_count >= self._max_reviews_per_restaurant:
                return

            page.evaluate(
                '(el) => el.scrollTop = el.scrollHeight', side_panel.element_handle()
            )
            page.wait_for_timeout(1000)

            review_divs = side_panel.locator(_REVIEW_SELECTOR)
            new_count = review_divs.count()

            _logger().debug(
                'Scrolled reviews panel, found %d reviews so far.', new_count
            )

            if new_count == review_divs_count:
                if retries_left > 0:
                    retries_left -= 1
                    continue
                break

            retries_left = self._reviews_scroll_retries
            review_divs_count = new_count

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

        return Restaurant(href=href, name=restaurant_name, basic_info=basic_info)

    # ----------------------------------------------------------------- Review helpers
    def _extract_review(self, review_div: Locator) -> Review | None:
        more_btn = review_div.locator('button.w8nwRe.kyuRq')
        if more_btn.count() > 0:
            try:
                more_btn.first.click(timeout=800)
                review_div.page.wait_for_timeout(120)
            except Exception:  # pylint: disable=broad-except
                pass

        rating = self._extract_rating(review_div.locator('span.kvMYJc').first)
        texts = self._extract_texts(review_div)
        texts.harmonize(self._normalize_text)

        if not self._has_meaningful_text(texts.translated):
            _logger().debug('Skipping review with no meaningful text: %s', texts.translated)
            return None

        if not self._long_enough(texts.translated):
            _logger().debug('Skipping review with too short text: %s', texts.translated)
            return None

        return Review(
            text=texts.translated.strip(),
            rating=rating,
            translated=texts.is_translated,
            original=texts.original.strip() if texts.is_translated else '',
        )

    def _extract_texts(self, review_div: Locator) -> ReviewTexts:
        text_spans = self._read_review_spans(review_div)
        translated_text = text_spans[0] if text_spans else ''

        dataset = review_div.evaluate('el => el.dataset || {}') or {}
        dataset_original = ''
        if isinstance(dataset, dict):
            dataset_original = (dataset.get('originalReviewText') or '').strip()

        has_marker = review_div.locator(_TRANSLATED_MARKER_SELECTOR).count() > 0
        is_translated = bool(has_marker or dataset_original)

        original_text = dataset_original
        if has_marker and not original_text:
            original_text = self._reveal_original(review_div)

        if not original_text and len(text_spans) > 1:
            original_text = text_spans[-1]

        texts = ReviewTexts(
            is_translated=is_translated,
            translated=translated_text,
            original=original_text,
        )
        return texts

    def _reveal_original(self, review_div: Locator) -> str:
        for selector in _SHOW_ORIGINAL_SELECTORS:
            button = review_div.locator(selector)
            if button.count() == 0:
                continue

            try:
                button.first.click(timeout=1000)
                review_div.page.wait_for_timeout(300)
                refreshed = self._read_review_spans(review_div)
                if refreshed:
                    return refreshed[-1]
            except Exception as exc:  # pylint: disable=broad-except
                _logger().debug('Failed to click "%s": %s', selector, exc)
        return ''

    def _read_review_spans(self, review_div: Locator) -> list[str]:
        spans = review_div.locator('span.wiI7pd').all_inner_texts()
        return [text.strip() for text in spans if text and text.strip()]

    def _extract_rating(self, stars_span: Locator) -> float:
        aria = stars_span.get_attribute('aria-label') or ''
        isolated_num = re.search(r'[\d]+', aria)
        return float(isolated_num.group(0)) if isolated_num else 0.0

    # ----------------------------------------------------------------- Text filters
    def _normalize_text(self, text: str) -> str:
        normalized = unicodedata.normalize('NFKC', text or '')
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized.casefold()

    def _has_meaningful_text(self, text: str) -> bool:
        if not text:
            return False
        has_letter = re.search(r'[A-Za-zÀ-ž]', text) is not None
        alnum_len = len(re.findall(r'[0-9A-Za-zÀ-ž]', text))
        return has_letter and alnum_len >= 3

    def _long_enough(self, text: str) -> bool:
        tokens = re.findall(r'\w+', text, flags=re.UNICODE)
        return len(tokens) >= 2 or len(self._normalize_text(text)) >= 10
