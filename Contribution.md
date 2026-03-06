This repository contains code used to scrape and preprocess textual data, train models and perform thorough evaluation and experiments. The repository uses the following tech stack:

- `Python`
- `Pytorch + Torch Lightning` - model building and training
- `MLFlow, Tensorboard` - experiment tracking, reproducibility


## Repository Structure

- `scripts/`: Entrypoints used directly to perform elements of the scraping/processing/training/evaluation pipeline 
    - `cfg/`: Hydra configuration files used by the scripts
- `src/`: Source code
    - `data/`: Resources related to data-related operations
        - `eda/`: Creating viz & stats for raw or processed data
        - `processing/`: Extracting numerical features from raw dataset
        - `scraping/`: Scraping raw data
        - `structs/`: Classes used to access directories of pre-defined structure
    - `model/`: Definition of the models and model components
    - `utils/`: Utilities used globally

## Code guidelines

1. Use guidelines defined in the Google Python Style Guide (https://google.github.io/styleguide/pyguide.html)
2. Avoid writing too long functions. If possible, split them into several ones.
3. Avoid using typedefs to define complex data structures. Make the most out of `Pydantic` or `dataclasses`.
4. Use docstrings for modules, classes and functions. For public functions, use Google docstring format. For private functions, use just a short description. Always in 3rd person.
5. Avoid using unnecessary `try-except` blocks and, in general, too nested code.
6. Every module should have its `_logger()` function, defining a logger as `logging.getLogger(__name__)`.
7. Do not use exceptions to handle unrecoverable problems. Use just a critical log and sys exit.
8. Do not use comments, unless absolutely necessary.
9. Do not use redundant variables, unless they contribute to the readability of the code. As a rule of thumb, if a variable is asigned a short expression, just use it directly.