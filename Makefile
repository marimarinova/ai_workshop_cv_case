PYTHON := python3
.PHONY: help install install-dev lint format format-check typecheck test compile clean

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode
	$(PYTHON) -m pip install -e .

install-dev: install ## Install package with dev dependencies
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff linter
	ruff check .

format: ## Run ruff formatter
	ruff format .

format-check: ## Check formatting without modifying files
	ruff format --check .

typecheck: ## Run mypy type checking
	mypy src/pickup_putdown

test: ## Run pytest
	$(PYTHON) -m pytest

compile: ## Compile all Python files to bytecode
	$(PYTHON) -m compileall src

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
