# Changelog

## 0.1.0

- Added the initial project configuration
- Introduced the scraping, processing and training pipelines
- Added scripts:
  - `scrape_google_reviews`
  - `process_dataset`
  - `perform_eda`
  - `train_model`
  - `summarize_experiment`
- Added v1.0 version of the paper

## 0.2.0

- Reworked the data scraping pipeline
  - modified the input configuration format
  - scraped a large raw dataset
- Introduced DVC
- Updated and enhanced EDA
- Reworked the data processing pipeline
  - introduced the train/test data processing separation
  - refactored modules responsible for fitting/processing the data
  - similified the extracted linguistic and numerical features
- Redesigned the model's architecture and training process
  - introduced the embedding sequence processing block
  - added only a single main training objective (regression or classification)
  - introduced the secondary translation prediction task
- Introduced more advanced evaluation utilities
  - enhanced experiments summary
  - added slice-based offline testing  