PYTHON := python3
.PHONY: install install-dev lint format typecheck test compile clean

install:
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

typecheck:
	mypy src/pickup_putdown

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall src

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
