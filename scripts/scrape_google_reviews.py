"""Script to scrape Google Maps reviews for locations matching specified queries."""

import dataclasses
import json
import logging
import pathlib
import sys

import hydra
import omegaconf
import tqdm  # type: ignore

from advanced_data_mining.data import maps_browser
from advanced_data_mining.utils import logging_utils
from advanced_data_mining.data import raw_ds


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _name_to_valid_path(name: str) -> str:
    """Make a string safe-ish to use in filenames."""
    keep = [c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name]
    return ''.join(keep).strip().replace(' ', '_')


def _scrape_reviews_for_restaurant(scraper: maps_browser.MapsBrowser,
                                   location: raw_ds.Restaurant,
                                   output_dir: pathlib.Path):
    """Scrapes reviews for a given restaurant and saves them to a file."""

    output_path = output_dir / f'{_name_to_valid_path(location.name)}.json'

    if output_path.exists():
        _logger().info(
            'Reviews already scraped for location: %s, skipping.', location.name
        )
        return

    _logger().info('Scraping reviews for location: %s', location.name)

    reviews = [dataclasses.asdict(review)
               for review
               in tqdm.tqdm(scraper.scrape_reviews_for(location), unit='review', desc='Reviews')]

    if not reviews:
        _logger().error('No reviews found for location: %s', location.name)
        return

    payload = {
        'location': dataclasses.asdict(location),
        'reviews': reviews
    }

    _logger().info(
        'Saving %d reviews to %s...',
        len(reviews),
        output_path,
    )

    with output_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


@hydra.main(version_base=None, config_path='cfg', config_name='scrape_google_reviews')
def main(script_cfg: omegaconf.DictConfig):
    """Scrapes Google Maps reviews for locations matching specified queries."""

    logging_utils.setup_logging(script_signature='scrape_google_reviews')

    if script_cfg.proxy is None:
        _logger().critical('Proxy configuration is required.')
        sys.exit(1)

    scraper = maps_browser.MapsBrowser(
        proxy_cfg={
            'server': script_cfg.proxy.server,
            'username': script_cfg.proxy.username,
            'password': script_cfg.proxy.password,
        },
        max_reviews_per_restaurant=script_cfg.max_reviews_per_restaurant,
    )

    output_dir = pathlib.Path(script_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for query in script_cfg.google_maps_queries:
        _logger().info('Starting scraping for query: %s', query)

        locations = scraper.get_locations_by_query(query)

        if not locations:
            _logger().warning('No locations found for query: %s', query)
            continue

        _logger().info('Found %d locations for query: %s', len(locations), query)

        query_output_dir = output_dir / _name_to_valid_path(query)

        for loc in locations:

            _scrape_reviews_for_restaurant(
                scraper=scraper,
                location=loc,
                output_dir=query_output_dir,
            )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
