# advanced-data-mining-project

The project provides tools for scraping restaurant textual reviews, which can be subsequently analysed using either traditional or advanced NLP approaches in order to get insight into relationship between written-form review and actual rating, helpfulness etc.

## Project structure

```yaml
.devcontainer   # Devcontainer setup.
scripts/        # Contains scripts for running scraping process, EDA etc.
src/            # Source code.
justfile        # Contains setup recipes, check out for installing deps etc.
pyproject.toml  # Core project configuration (version, dependencies, dev deps).
```

## Example usage

```
just setup_env
just install_deps
uv run python scripts/scrape_google_reviews.py
```
