"""Module that extracts reviews from Google Maps page corresponding to specific restaurant."""

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional, Tuple

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
    """Performs Unicode normalization."""
    normalized = unicodedata.normalize('NFKC', text or '')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized.casefold()


def _has_meaningful_text(text: str) -> bool:
    """Tells if the text contain any content worth processing."""
    if not text:
        return False
    has_letter = re.search(r'[A-Za-zÀ-ž]', text) is not None
    alnum_len = len(re.findall(r'[0-9A-Za-zÀ-ž]', text))
    return has_letter and alnum_len >= 3


def _has_at_least_one_word(text: str) -> bool:
    """Tells if the text contains at least one word."""
    tokens = re.findall(r'\w+', text, flags=re.UNICODE)
    return len(tokens) >= 1


class ReviewsExtractor:
    """Extracts reviews from Google Maps restaurant page."""

    _REVIEWS_SCROLL_RETRIES = 5
    _ITERATION_LOG_INTERVAL = 20

    @classmethod
    async def create(cls, page: Page, max_reviews: int) -> 'ReviewsExtractor':
        """Spawns the extractor and prepares the page for review extraction."""

        if not await ReviewsExtractor._open_more_reviews(page):
            return ReviewsExtractor(None, max_reviews)

        await ReviewsExtractor._scroll_reviews_to_end(page, max_reviews)

        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR).first
        review_divs = side_panel.locator(_REVIEW_SELECTOR)

        return ReviewsExtractor(review_divs, max_reviews)

    def __init__(self, review_divs: Optional[Locator], max_reviews: int) -> None:

        self._review_divs = review_divs
        self._max_reviews = max_reviews

    async def get_n_reviews(self) -> int:
        """Returns the number of reviews extracted from the page."""
        if self._review_divs is None:
            return 0

        return await self._review_divs.count()

    async def iter_reviews(self) -> AsyncIterator[Review]:
        """Yields the extracted reviews one by one."""

        if self._review_divs is None:
            return

        n_reviews_to_take = min(await self._review_divs.count(), self._max_reviews)

        for i in range(n_reviews_to_take):

            if (i + 1) % self._ITERATION_LOG_INTERVAL == 0:
                _logger().debug('Extracted %d reviews.', i + 1)

            review_div = self._review_divs.nth(i)

            try:
                review = await self._extract_review(review_div)
                if review is not None:
                    yield review

            except Exception as exc:  # pylint: disable=broad-except
                _logger().error('Failed to extract review: %s', exc)

    @staticmethod
    async def _open_more_reviews(page: Page) -> bool:
        """Opens the scrollable reviews panel to enable scraping all reviews."""

        tab_buttons = page.locator('button.hh2c6').filter(has_text='Reviews')
        buttons_count = await tab_buttons.count()

        if buttons_count != 1:
            _logger().debug('Couldn\'t locate the "Reviews" button on side panel!')

        else:

            try:
                await tab_buttons.first.click(timeout=4000)
                await page.wait_for_timeout(2000)

                return True

            except Exception as e:  # pylint: disable=broad-except
                _logger().error('Could\'t open the reviews panel!: %s', e)

        more_reviews_btn = page.locator('button:has-text("More reviews")')

        if await more_reviews_btn.count() == 0:
            _logger().debug('Couldn\'t locate the "More reviews" button!')
        
        else:
            try:
                await more_reviews_btn.first.click(timeout=4000)
                await page.wait_for_timeout(2000)

                return True

            except Exception as exc:  # pylint: disable=broad-except
                _logger().debug('Failed to click More reviews button: %s', exc)

        return False

    @staticmethod
    async def _scroll_reviews_to_end(page: Page, max_reviews: int) -> None:
        """Scrolls the review panel until new reviews stop loading in reasonable time.

        If the number of loaded reviews reaches `max_reviews`, scrolling stops earlier.
        """

        side_panel = page.locator(_REVIEWS_CONTAINER_SELECTOR)

        try:
            await side_panel.wait_for()
        except Exception:  # pylint: disable=broad-except
            _logger().error('Reviews side panel did not load in time!')
            return

        review_divs = side_panel.first.locator(_REVIEW_SELECTOR)

        try:
            await review_divs.first.wait_for()
        except Exception:  # pylint: disable=broad-except
            _logger().error('Review divs did not load in time!')
            return

        review_divs_count = await review_divs.count()
        retries_left = ReviewsExtractor._REVIEWS_SCROLL_RETRIES

        while retries_left > 0:
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

            if new_count <= review_divs_count:
                retries_left -= 1

            else:
                retries_left = ReviewsExtractor._REVIEWS_SCROLL_RETRIES
                review_divs_count = new_count

    async def _extract_review(self, review_div: Locator) -> Review | None:
        """Extracts a single review from its div element."""

        more_btn = review_div.locator('button.w8nwRe.kyuRq')
        if await more_btn.count() > 0:
            try:
                await more_btn.first.click(timeout=800)
                await review_div.page.wait_for_timeout(120)
            except TimeoutError:  # pylint: disable=broad-except
                pass

        rating = await self._extract_rating(review_div.locator('span.kvMYJc').first)
        translated_txt, original_txt = await self._extract_main_review_texts(review_div)

        if not _has_meaningful_text(translated_txt):
            _logger().debug('Skipping review with no meaningful text: %s', translated_txt)
            return None

        if not _has_at_least_one_word(translated_txt):
            _logger().debug('Skipping review with too short text: %s', translated_txt)
            return None

        return Review(
            text=translated_txt.strip(),
            rating=rating,
            original=original_txt.strip() if original_txt else None,
            author=await self._extract_author(review_div),
            categorized_opinions=await self._extract_categorized_opinions(review_div)
        )

    async def _extract_main_review_texts(self, review_div: Locator) -> Tuple[str, str | None]:
        """Extracts the main translated and original review texts from a review div."""

        text_spans = await self._read_review_spans(review_div)

        translated_text = text_spans[0] if text_spans else ''
        original_text = None

        dataset = await review_div.evaluate('el => el.dataset || {}') or {}
        if isinstance(dataset, dict):
            original_text = (dataset.get('originalReviewText') or '').strip()

        has_marker = await review_div.locator(_TRANSLATED_MARKER_SELECTOR).count() > 0

        if not original_text and has_marker:
            original_text = await self._reveal_original(review_div)

        if not original_text and len(text_spans) > 1:
            original_text = text_spans[-1]

        if _normalize_text(translated_text) == _normalize_text(original_text or ''):
            original_text = None

        return translated_text, original_text

    async def _extract_author(self, review_div: Locator) -> Author:
        """Extracts the author information from a review div."""

        name_locator = review_div.locator(_AUTHOR_NAME_SELECTOR)
        stats_locator = review_div.locator(_AUTHOR_STATS_SELECTOR)

        if await name_locator.count() == 0 or await stats_locator.count() == 0:
            _logger().warning('Author information not found in review.')
            return Author(name='Anonymous', n_reviews=0)

        name = await name_locator.first.inner_text()
        stats = await stats_locator.first.inner_text()

        match = re.search(r'(\d+)\s+reviews?', stats)
        n_reviews = int(match.group(1)) if match else None

        return Author(
            name=name.strip(),
            n_reviews=n_reviews,
        )

    async def _extract_categorized_opinions(self, review_div: Locator) -> Optional[Dict[str, str]]:
        """Extracts the listed categorized opinions from a review div, if available.

        Categorized opinions are formatted texts placed below the main review text,
        usually in the form of "Food: Excellent", "Service: Poor", etc.
        """

        category_divs = review_div.locator('div.PBK6be')

        n_categories = await category_divs.count()

        if n_categories == 0:
            return None

        categorized_opinions: Dict[str, str] = {}

        for i in range(n_categories):
            kv_spans = category_divs.nth(i).locator('span.RfDO5c')

            if await kv_spans.count() != 2:
                continue

            key = await kv_spans.nth(0).inner_text()
            value = await kv_spans.nth(1).inner_text()

            categorized_opinions[key.strip()] = value.strip()

        return categorized_opinions

    async def _reveal_original(self, review_div: Locator) -> str:
        """Triggers the "Show original" button to reveal the original review text."""

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
        """Reads all textual spans from the main review text area."""

        spans = await review_div.locator('span.wiI7pd').all_inner_texts()
        return [text.strip() for text in spans if text and text.strip()]

    async def _extract_rating(self, stars_span: Locator) -> float:
        """Extracts the rating value from the stars span element."""

        aria = await stars_span.get_attribute('aria-label') or ''
        isolated_num = re.search(r'[\d]+', aria)
        return float(isolated_num.group(0)) if isolated_num else 0.0
