"""Module that extracts reviews from Google Maps page corresponding to specific restaurant."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import AsyncIterator

from playwright.async_api import Locator
from playwright.async_api import Page

from advanced_data_mining.data.raw_ds import Review, Author


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


_REVIEWS_CONTAINER_SELECTOR = 'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde'
_REVIEW_SELECTOR = 'div.jftiEf'
_SHOW_ORIGINAL_SELECTORS = (
    'button:has-text("See original")',
    'button:has-text("Show original")',
)
_TRANSLATED_MARKER_SELECTOR = 'span:has-text("Translated by Google")'
_AUTHOR_NAME_SELECTOR = 'button.al6Kxe div.d4r55'
_AUTHOR_STATS_SELECTOR = 'button.al6Kxe div.RfnDt'


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

    _REVIEWS_SCROLL_RETRIES = 5

    @classmethod
    async def create(cls, page: Page, max_reviews: int) -> 'ReviewsExtractor':
        """Spawns the extractor and prepares the page for review extraction."""

        await ReviewsExtractor._open_more_reviews(page)
        await ReviewsExtractor._scroll_reviews_to_end(page, max_reviews)

        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR).first
        review_divs = side_panel.locator(_REVIEW_SELECTOR)

        return ReviewsExtractor(review_divs, max_reviews)

    def __init__(self, review_divs: Locator, max_reviews: int) -> None:

        self._review_divs = review_divs
        self._max_reviews = max_reviews

    async def get_n_reviews(self) -> int:
        """Returns the number of reviews extracted from the page."""
        return await self._review_divs.count()

    async def iter_reviews(self) -> AsyncIterator[Review]:
        """Yields the extracted reviews one by one."""

        n_reviews_to_take = min(await self._review_divs.count(), self._max_reviews)

        for i in range(n_reviews_to_take):
            review_div = self._review_divs.nth(i)

            try:
                review = await self._extract_review(review_div)
                if review is not None:
                    yield review

            except Exception as exc:  # pylint: disable=broad-except
                _logger().error('Failed to extract review: %s', exc)

    @staticmethod
    async def _open_more_reviews(page: Page) -> None:
        button = page.locator('button:has-text("More reviews")')
        if await button.count() == 0:
            return
        try:
            await button.first.click(timeout=4000)
            await page.wait_for_timeout(2000)
        except Exception as exc:  # pylint: disable=broad-except
            _logger().debug('Failed to click More reviews button: %s', exc)

    @staticmethod
    async def _scroll_reviews_to_end(page: Page, max_reviews: int) -> None:
        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR)
        if await side_panel.count() == 0:
            _logger().critical('Cannot find reviews side panel!')
            return

        review_divs = side_panel.first.locator(_REVIEW_SELECTOR)
        await page.wait_for_timeout(1000)
        review_divs_count = await review_divs.count()

        if review_divs_count == 0:
            _logger().warning('No reviews found in the side panel!')
            return

        retries_left = ReviewsExtractor._REVIEWS_SCROLL_RETRIES

        while review_divs_count > 0:
            if review_divs_count >= max_reviews:
                return

            await page.evaluate(
                '(el) => el.scrollTop = el.scrollHeight', await side_panel.element_handle()
            )
            await page.wait_for_timeout(1000)

            review_divs = side_panel.first.locator(_REVIEW_SELECTOR)
            new_count = await review_divs.count()

            _logger().debug(
                'Scrolled reviews panel, found %d reviews so far.', new_count
            )

            if new_count == review_divs_count:
                if retries_left > 0:
                    retries_left -= 1
                    continue
                break

            retries_left = ReviewsExtractor._REVIEWS_SCROLL_RETRIES
            review_divs_count = new_count

    async def _extract_review(self, review_div: Locator) -> Review | None:
        more_btn = review_div.locator('button.w8nwRe.kyuRq')
        if await more_btn.count() > 0:
            try:
                await more_btn.first.click(timeout=800)
                await review_div.page.wait_for_timeout(120)
            except TimeoutError:  # pylint: disable=broad-except
                pass

        rating = await self._extract_rating(review_div.locator('span.kvMYJc').first)
        texts = await self._extract_texts(review_div)

        if not _has_meaningful_text(texts.translated):
            _logger().debug('Skipping review with no meaningful text: %s', texts.translated)
            return None

        if not _long_enough(texts.translated):
            _logger().debug('Skipping review with too short text: %s', texts.translated)
            return None

        return Review(
            text=texts.translated.strip(),
            rating=rating,
            original=texts.original.strip() if texts.is_translated else None,
            author=await self._extract_author(review_div),
        )

    async def _extract_texts(self, review_div: Locator) -> ReviewTexts:
        text_spans = await self._read_review_spans(review_div)
        translated_text = text_spans[0] if text_spans else ''

        dataset = await review_div.evaluate('el => el.dataset || {}') or {}
        dataset_original = ''
        if isinstance(dataset, dict):
            dataset_original = (dataset.get('originalReviewText') or '').strip()

        has_marker = await review_div.locator(_TRANSLATED_MARKER_SELECTOR).count() > 0
        is_translated = bool(has_marker or dataset_original)

        original_text = dataset_original
        if has_marker and not original_text:
            original_text = await self._reveal_original(review_div)

        if not original_text and len(text_spans) > 1:
            original_text = text_spans[-1]

        texts = ReviewTexts(
            is_translated=is_translated,
            translated=translated_text,
            original=original_text,
        )
        return texts

    async def _extract_author(self, review_div: Locator) -> Author:

        name_locator = review_div.locator(_AUTHOR_NAME_SELECTOR)
        stats_locator = review_div.locator(_AUTHOR_STATS_SELECTOR)

        if await name_locator.count() == 0 or await stats_locator.count() == 0:
            _logger().warning('Author information not found in review.')
            return Author(name='Anonymous', n_reviews=0)

        name = await name_locator.first.inner_text()
        stats = await stats_locator.first.inner_text()

        match = re.match(r'(\d+)\s+reviews?', stats)
        n_reviews = int(match.group(1)) if match else None

        return Author(
            name=name.strip(),
            n_reviews=n_reviews,
        )

    async def _reveal_original(self, review_div: Locator) -> str:
        for selector in _SHOW_ORIGINAL_SELECTORS:
            button = review_div.locator(selector)
            if await button.count() == 0:
                continue

            try:
                await button.first.click(timeout=1000)
                await review_div.page.wait_for_timeout(300)
                refreshed = await self._read_review_spans(review_div)
                if refreshed:
                    return refreshed[-1]
            except Exception as exc:  # pylint: disable=broad-except
                _logger().debug('Failed to click "%s": %s', selector, exc)
        return ''

    async def _read_review_spans(self, review_div: Locator) -> list[str]:
        spans = await review_div.locator('span.wiI7pd').all_inner_texts()
        return [text.strip() for text in spans if text and text.strip()]

    async def _extract_rating(self, stars_span: Locator) -> float:
        aria = await stars_span.get_attribute('aria-label') or ''
        isolated_num = re.search(r'[\d]+', aria)
        return float(isolated_num.group(0)) if isolated_num else 0.0
