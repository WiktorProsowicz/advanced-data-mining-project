set positional-arguments

# Sets up a Python virtual environment.
setup_env:
    @echo "Setting up virtual environment..."
    uv venv -p /opt/conda/bin/python --system-site-packages

# Install project package, Python libs, browser engine etc.
build_project:
    #!/usr/bin/env bash
    echo "Installing project..."
    source .venv/bin/activate
    uv pip install -e .
    uv run playwright install --with-deps firefox
    
install_dev_deps:
    @echo "Installing development dependencies..."
    uv pip install -r pyproject.toml --extra dev