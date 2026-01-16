"""Script to scrape Google Maps reviews for locations matching specified queries."""

import json
import logging
import pathlib
import sys
from typing import List
import asyncio

import hydra
import omegaconf
from playwright.sync_api import sync_playwright, ProxySettings as SyncProxySettings
from playwright.async_api import async_playwright, ProxySettings as AsyncProxySettings
from playwright.async_api import BrowserContext as AsyncBrowserContext

from advanced_data_mining.data.scraping import maps_browser
from advanced_data_mining.data import raw_ds
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger('advanced_data_mining')


def _name_to_valid_path(name: str) -> str:
    """Make a string safe-ish to use in filenames."""
    keep = [c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name]
    return ''.join(keep).strip().replace(' ', '_')


async def _scrape_reviews_for_restaurant(scraper: maps_browser.MapsBrowser,
                                         location: raw_ds.Restaurant,
                                         output_dir: pathlib.Path,
                                         browser_context: AsyncBrowserContext) -> None:
    """Scrapes reviews for a given restaurant."""

    output_path = output_dir / f'{_name_to_valid_path(location.name)}.json'

    if output_path.exists():
        _logger().info(
            'Reviews already scraped for location: %s, skipping.', location.name
        )
        return

    _logger().info('Scraping reviews for location: %s', location.name)

    page = await browser_context.new_page()
    page.set_default_timeout(40000)

    reviews = [review.model_dump()
               async for review
               in scraper.scrape_reviews_for(location, page)]

    if not reviews:
        _logger().error('No reviews found for location: %s', location.name)
        return

    payload = {
        'location': location.model_dump(),
        'reviews': reviews
    }

    _logger().info(
        'Saving %d reviews to %s...',
        len(reviews),
        output_path,
    )

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


async def _scrape_reviews_for_restaurants(scraper: maps_browser.MapsBrowser,
                                          proxy_cfg: AsyncProxySettings,
                                          locations: List[raw_ds.Restaurant],
                                          output_dir: pathlib.Path) -> None:
    """Scrapes reviews for a given restaurant and saves them to a file."""

    async with async_playwright() as async_pw:
        browser = await async_pw.firefox.launch(
            headless=True,
            proxy=proxy_cfg
        )
        browser_context = await browser.new_context(
            locale='en-US',
            extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
        )

        tasks = [asyncio.create_task(_scrape_reviews_for_restaurant(scraper,
                                                                    location,
                                                                    output_dir,
                                                                    browser_context))
                 for location in locations]

        await asyncio.gather(*tasks)


@hydra.main(version_base=None, config_path='cfg', config_name='scrape_google_reviews')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Scrapes Google Maps reviews for locations matching specified queries."""

    logging_utils.setup_logging(script_signature='scrape_google_reviews')

    _logger().info('Script started with configuration: %s', omegaconf.OmegaConf.to_yaml(script_cfg))

    if script_cfg.proxy is None:
        _logger().critical('Proxy configuration is required.')
        sys.exit(1)

    scraper = maps_browser.MapsBrowser(
        max_reviews_per_restaurant=script_cfg.max_reviews_per_restaurant,
        max_restaurants_per_location=script_cfg.max_restaurants_per_location,
    )

    output_dir = pathlib.Path(script_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    location_pairs = [(primary_loc, secondary_loc)
                      for primary_loc, secondary_locs in script_cfg.google_maps_queries.items()
                      for secondary_loc in secondary_locs]

    for primary_loc, secondary_loc in location_pairs:
        _logger().info('Starting scraping for location: %s %s', primary_loc, secondary_loc)

        with sync_playwright() as sync_pw:
            browser = sync_pw.firefox.launch(
                headless=True,
                proxy=SyncProxySettings(
                    server=script_cfg.proxy.server,
                    username=script_cfg.proxy.get('username'),
                    password=script_cfg.proxy.get('password'),
                ),
            )
            page = browser.new_context(
                locale='en-US',
                extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'},
            ).new_page()

            locations = list(scraper.get_locations_by_query(primary_loc, secondary_loc, page))

            browser.close()

        if not locations:
            _logger().warning('No locations found for location: %s %s', primary_loc, secondary_loc)
            continue

        _logger().info('Found %d locations for location: %s %s',
                       len(locations), primary_loc, secondary_loc)

        query_output_dir = (output_dir /
                            _name_to_valid_path(primary_loc) /
                            _name_to_valid_path(secondary_loc))
        query_output_dir.mkdir(parents=True, exist_ok=True)

        asyncio.run(_scrape_reviews_for_restaurants(
            scraper=scraper,
            proxy_cfg=AsyncProxySettings(
                server=script_cfg.proxy.server,
                username=script_cfg.proxy.get('username'),
                password=script_cfg.proxy.get('password'),
            ),
            locations=locations,
            output_dir=query_output_dir,
        ))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
