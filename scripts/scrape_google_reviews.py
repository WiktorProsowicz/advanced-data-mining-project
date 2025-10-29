# -*- coding: utf-8 -*-
"""Script to scrape Google Maps reviews for locations matching specified queries."""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
import omegaconf
import tqdm  # type: ignore

from advanced_data_mining.data import maps_browser
from advanced_data_mining.utils import logging_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


# --- Helpers -----------------------------------------------------------------
ORIGINAL_KEYS = ("original", "original_text", "text_original")
TRANSLATED_KEYS = ("translated_text", "text", "text_translated")
TRANSLATED_BOOL_KEYS = ("translated", "is_translated", "was_translated")


def _first_present(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    """Return first non-None value for any of the given keys."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def normalize_review(review_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    If original and translated texts are identical (ignoring surrounding whitespace),
    set translated flag to False and blank out the original field.
    Works defensively with multiple common key names.
    """
    original_val = _first_present(review_dict, ORIGINAL_KEYS)
    translated_val = _first_present(review_dict, TRANSLATED_KEYS)

    if isinstance(original_val, str) and isinstance(translated_val, str):
        if original_val.strip() == translated_val.strip():
            # Flip any known boolean 'translated' flags to False
            for bkey in TRANSLATED_BOOL_KEYS:
                if bkey in review_dict:
                    review_dict[bkey] = False
            # Blank all known 'original' text fields that are present
            for okey in ORIGINAL_KEYS:
                if okey in review_dict:
                    review_dict[okey] = ""

    return review_dict


def sanitize_for_fs(name: str) -> str:
    """Make a string safe-ish to use in filenames."""
    keep = [c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name]
    return "".join(keep).strip().replace(" ", "_")


# --- Main --------------------------------------------------------------------
@hydra.main(version_base=None, config_path="cfg", config_name="scrape_google_reviews")
def main(script_cfg: omegaconf.DictConfig):
    """Scrapes Google Maps reviews for locations matching specified queries."""
    logging_utils.setup_logging(script_signature="scrape_google_reviews")

    if script_cfg.proxy is None:
        _logger().critical("Proxy configuration is required.")
        return

    proxy_cfg = {
        "server": script_cfg.proxy.server,
        "username": script_cfg.proxy.username,
        "password": script_cfg.proxy.password,
    }

    scraper = maps_browser.MapsBrowser(
        proxy_cfg=proxy_cfg,
        max_reviews_per_restaurant=script_cfg.max_reviews_per_restaurant,
    )

    output_dir = Path(script_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for query in script_cfg.google_maps_queries:
        _logger().info("Starting scraping for query: %s", query)

        locations = scraper.get_locations_by_query(query)
        if not locations:
            _logger().warning("No locations found for query: %s", query)
            continue

        _logger().info("Found %d locations for query: %s", len(locations), query)

        for loc in locations:
            _logger().info("Scraping reviews for location: %s", loc.name)

            output_path = (
                output_dir
                / f"reviews_{sanitize_for_fs(query)}_{sanitize_for_fs(loc.name)}.json"
            )

            if output_path.exists():
                _logger().info(
                    "Reviews already scraped for location: %s, skipping.", loc.name
                )
                continue

            # Scrape -> dataclass -> dict -> normalize
            raw_iter = scraper.scrape_reviews_for(loc)

            normalized_reviews: List[Dict[str, Any]] = []
            changed_count = 0

            for review in tqdm.tqdm(raw_iter, unit="review", desc="Reviews"):
                rd = dataclasses.asdict(review)
                before = (rd.get("original_text"), rd.get("translated_text"), rd.get("translated"))
                rd = normalize_review(rd)
                after = (rd.get("original_text"), rd.get("translated_text"), rd.get("translated"))
                if before != after:
                    changed_count += 1
                normalized_reviews.append(rd)

            if not normalized_reviews:
                _logger().error("No reviews found for location: %s", loc.name)
                continue

            payload = {
                "location": dataclasses.asdict(loc),
                "reviews": normalized_reviews,
                "query": query,
            }

            output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=4), encoding="utf-8")
            _logger().info(
                "Saved %d reviews to %s (normalized %d).",
                len(normalized_reviews),
                output_path,
                changed_count,
            )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
