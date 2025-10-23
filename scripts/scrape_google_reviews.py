# -*- coding: utf-8 -*-
"""Script to scrape Google Maps reviews for locations matching specified queries."""
import dataclasses
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List

import hydra
import omegaconf
import tqdm  # type: ignore

from advanced_data_mining.data import maps_browser
from advanced_data_mining.utils import logging_utils


def _logger():
    return logging.getLogger(__name__)


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
        max_reviews_per_restaurant=script_cfg.max_reviews_per_restaurant)

    os.makedirs(script_cfg.output_dir, exist_ok=True)

    for query in script_cfg.google_maps_queries:

        _logger().info('Starting scraping for query: %s', query)

        locations = scraper.get_locations_by_query(query)

        if not locations:
            _logger().warning('No locations found for query: %s', query)
            continue

        _logger().info('Found %d locations for query: %s',
                       len(locations), query)

        for loc in locations:

            _logger().info('Scraping reviews for location: %s', loc.name)

            output_path = os.path.join(
                script_cfg.output_dir,
                f"reviews_{query.replace(' ', '_')}_{loc.name.replace(' ', '_')}.json"
            )

            if os.path.exists(output_path):
                _logger().info('Reviews already scraped for location: %s, skipping.', loc.name)
                continue

            reviews: List[Dict[str, Any]] = []

            for review in tqdm.tqdm(scraper.scrape_reviews_for(loc),
                                    unit='review',
                                    desc='Reviews'):

                reviews.append(dataclasses.asdict(review))

            if not reviews:
                _logger().error('No reviews found for location: %s', loc.name)
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({'location': dataclasses.asdict(loc),
                           'reviews': reviews,
                           'query': query},
                          f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
