"""Script to scrape Google Maps reviews for locations matching specified queries."""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import hydra
import omegaconf
import tqdm  # type: ignore

from advanced_data_mining.data import maps_browser
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def sanitize_for_fs(name: str) -> str:
    """Make a string safe-ish to use in filenames."""
    keep = [c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name]
    return ''.join(keep).strip().replace(' ', '_')


# --- Main --------------------------------------------------------------------
@hydra.main(version_base=None, config_path='cfg', config_name='scrape_google_reviews')
def main(script_cfg: omegaconf.DictConfig):
    """Scrapes Google Maps reviews for locations matching specified queries."""
    logging_utils.setup_logging(script_signature='scrape_google_reviews')

    if script_cfg.proxy is None:
        _logger().critical('Proxy configuration is required.')
        return

    proxy_cfg = {
        'server': script_cfg.proxy.server,
        'username': script_cfg.proxy.username,
        'password': script_cfg.proxy.password,
    }

    scraper = maps_browser.MapsBrowser(
        proxy_cfg=proxy_cfg,
        max_reviews_per_restaurant=script_cfg.max_reviews_per_restaurant,
    )

    output_dir = Path(script_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for query in script_cfg.google_maps_queries:
        _logger().info('Starting scraping for query: %s', query)

        locations = scraper.get_locations_by_query(query)
        if not locations:
            _logger().warning('No locations found for query: %s', query)
            continue

        _logger().info('Found %d locations for query: %s', len(locations), query)

        for loc in locations:
            _logger().info('Scraping reviews for location: %s', loc.name)

            output_path = (
                output_dir
                / f'reviews_{sanitize_for_fs(query)}_{sanitize_for_fs(loc.name)}.json'
            )

            if output_path.exists():
                _logger().info(
                    'Reviews already scraped for location: %s, skipping.', loc.name
                )
                continue

            reviews: list[dict[str, Any]] = []

            for review in tqdm.tqdm(scraper.scrape_reviews_for(loc), unit='review', desc='Reviews'):
                reviews.append(dataclasses.asdict(review))

            if not reviews:
                _logger().error('No reviews found for location: %s', loc.name)
                continue

            payload = {
                'location': dataclasses.asdict(loc),
                'reviews': reviews,
                'query': query,
            }

            _logger().info(
                'Saving %d reviews to %s...',
                len(reviews),
                output_path,
            )

            with output_path.open('w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
