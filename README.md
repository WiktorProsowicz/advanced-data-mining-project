# advanced-data-mining-project

The project provides tools for scraping restaurant textual reviews, which can be subsequently analysed using either traditional or advanced NLP approaches in order to get insight into relationship between written-form review and actual rating, helpfulness etc.

We recommend you to check out the **paper** describing the theory, methods and experiments related to the project. You can find it in the `doc` directory.

## Project structure

```yaml
- .devcontainer   # Devcontainer setup.
- doc/            # Documentation, experiments description.         
- scripts/        # Contains scripts for running scraping process, EDA etc.
- src/            # Source code.
- justfile        # Contains setup recipes, check out for installing deps etc.
- pyproject.toml  # Core project configuration (version, dependencies, dev deps).
```

## Usage

The project's public API, available for the user, can be found in the `scripts` directory. It contains scripts for running data scraping & processing pipelines, models training and obtaining visualizations and summaries. 

```
just setup_env
just build_project
uv run python scripts/scrape_google_reviews.py +proxy.server=SERVER +proxy.username=USER +proxy.password=PASSWORD
```


## Changelog

The changes made to the project are recorded to the `Changelog.md` file.