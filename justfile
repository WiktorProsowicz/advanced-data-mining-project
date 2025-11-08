set positional-arguments

# Sets up a Python virtual environment.
setup_env:
    @echo "Setting up virtual environment..."
    python3.11 -m venv .venv --system-site-packages
    echo 'export MPLBACKEND=Agg' >> .venv/bin/activate

# Install project package, Python libs, browser engine etc.
build_project:
    #!/usr/bin/env bash
    echo "Installing project..."
    source .venv/bin/activate
    pip install -e .
    playwright install --with-deps firefox

# Install development dependencies.
install_dev_deps:
    #!/usr/bin/env bash
    echo "Installing development dependencies..."
    source .venv/bin/activate
    pip install -e .[dev]

# Run static code checks using pre-commit.
run_static_checks:
    #!/usr/bin/env bash
    echo "Running static checks..."
    source .venv/bin/activate
    pre-commit run --all-files

# Run an arbitrary python script inside the virtual environment.
run_python *args:
    #!/usr/bin/env bash
    source .venv/bin/activate
    python "$@"
