.PHONY: install test lint format clean

install:
    pip install -e ".[dev]"

test:
    pytest tests/ -v --cov=src

test-fast:
    pytest tests/unit/ -v -x

lint:
    flake8 src/ scripts/ tests/

format:
    black src/ scripts/ tests/

clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -name "*.pyc" -delete
