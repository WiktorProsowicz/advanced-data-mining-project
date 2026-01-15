"""Module that extracts reviews from Google Maps page corresponding to specific restaurant."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator

from playwright.sync_api import Locator
from playwright.sync_api import Page

from advanced_data_mining.data.raw_ds import Review


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


_REVIEWS_CONTAINER_SELECTOR = 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde'
_REVIEW_SELECTOR = 'div.jftiEf'
_SHOW_ORIGINAL_SELECTORS = (
    'button:has-text("See original")',
    'button:has-text("Show original")',
)
_TRANSLATED_MARKER_SELECTOR = 'span:has-text("Translated by Google")'


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text or '')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized.casefold()


def _has_meaningful_text(text: str) -> bool:
    if not text:
        return False
    has_letter = re.search(r'[A-Za-zÀ-ž]', text) is not None
    alnum_len = len(re.findall(r'[0-9A-Za-zÀ-ž]', text))
    return has_letter and alnum_len >= 3


def _long_enough(text: str) -> bool:
    tokens = re.findall(r'\w+', text, flags=re.UNICODE)
    return len(tokens) >= 2 or len(_normalize_text(text)) >= 10


@dataclass
class ReviewTexts:
    """Holds the textual fragments extracted from a review block.

    The `is_translated` flag is guaranteed to be consistent with the presence of the
    `original` text, i.e. the original is not empty
    """

    is_translated: bool
    translated: str
    original: str

    def __init__(self, is_translated: bool, translated: str, original: str) -> None:

        self.is_translated = is_translated
        self.translated = translated
        self.original = original

        if not is_translated:
            self.original = ''

        elif _normalize_text(self.translated) == _normalize_text(self.original):
            self.is_translated = False
            self.original = ''


class ReviewsExtractor:
    """Extracts reviews from Google Maps restaurant page."""

    def __init__(self, page: Page, max_reviews: int) -> None:

        self._max_reviews = max_reviews
        self._reviews_scroll_retries = 5

        self._open_more_reviews(page)
        self._scroll_reviews_to_end(page)

        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR).first
        self._review_divs = side_panel.locator(_REVIEW_SELECTOR)

    @property
    def n_reviews(self) -> int:
        """Returns the number of reviews extracted from the page."""
        return self._review_divs.count()

    def iter_reviews(self) -> Iterator[Review]:
        """Yields the extracted reviews one by one."""

        n_reviews_to_take = min(self._review_divs.count(), self._max_reviews)

        for i in range(n_reviews_to_take):
            review_div = self._review_divs.nth(i)
            try:
                review = self._extract_review(review_div)
                if review is not None:
                    yield review
            except Exception as exc:  # pylint: disable=broad-except
                _logger().error('Failed to extract review: %s', exc)

    def _open_more_reviews(self, page: Page) -> None:
        button = page.locator('button:has-text("More reviews")')
        if button.count() == 0:
            return
        try:
            button.first.click(timeout=4000)
            page.wait_for_timeout(2000)
        except Exception as exc:  # pylint: disable=broad-except
            _logger().debug('Failed to click More reviews button: %s', exc)

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
            if review_divs_count >= self._max_reviews:
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

    def _extract_review(self, review_div: Locator) -> Review | None:
        more_btn = review_div.locator('button.w8nwRe.kyuRq')
        if more_btn.count() > 0:
            try:
                more_btn.first.click(timeout=800)
                review_div.page.wait_for_timeout(120)
            except TimeoutError:  # pylint: disable=broad-except
                pass

        rating = self._extract_rating(review_div.locator('span.kvMYJc').first)
        texts = self._extract_texts(review_div)

        if not _has_meaningful_text(texts.translated):
            _logger().debug('Skipping review with no meaningful text: %s', texts.translated)
            return None

        if not _long_enough(texts.translated):
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
