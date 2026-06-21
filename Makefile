PYTHON := python3
.PHONY: help install install-dev lint format format-check typecheck test compile clean env-ingest ingest env-setup

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

env-ingest: ## Prompt for S3 credentials and run ingest with them set in the environment
	@read -rp "AWS Access Key ID: " ACCESS_KEY && \
	read -p "AWS Secret Access Key: " SECRET_KEY && echo && \
	AWS_ACCESS_KEY_ID="$$ACCESS_KEY" \
	AWS_SECRET_ACCESS_KEY="$$SECRET_KEY" \
	pickup-putdown --config configs/storage.yaml

ingest: ## Run ingest using credentials already set in the environment
	pickup-putdown --config configs/storage.yaml

env-setup: ## Prompt for S3 credentials and print export commands to source in your shell
	@read -rp "AWS Access Key ID: " ACCESS_KEY && \
	read -p "AWS Secret Access Key: " SECRET_KEY && \
	read -rp "S3 Bucket URI (s3://bucket/prefix): " BUCKET_URI && \
	read -rp "S3 Region (e.g. us-east-1): " REGION && \
	read -rp "Endpoint URL (leave empty for AWS default): " ENDPOINT && \
	read -rp "Anonymous access (y/N): " ANON && \
	echo && \
	echo "Run the following commands to set up your environment:" && \
	echo && \
	echo "export AWS_ACCESS_KEY_ID=$$ACCESS_KEY" && \
	echo "export AWS_SECRET_ACCESS_KEY=$$SECRET_KEY" && \
	echo "export PICKUP_PUTDOWN_STORAGE_BUCKET_URI=$$BUCKET_URI" && \
	echo "export PICKUP_PUTDOWN_STORAGE_REGION=$$REGION" && \
	if [ -n "$$ENDPOINT" ]; then echo "export PICKUP_PUTDOWN_STORAGE_ENDPOINT_URL=$$ENDPOINT"; fi && \
	if [ "$$ANON" = "y" ] || [ "$$ANON" = "Y" ]; then echo "export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=true"; fi && \
	echo && \
	echo "Or copy-paste the block below:" && \
	echo "----------------------------------------" && \
	echo "export AWS_ACCESS_KEY_ID=$$ACCESS_KEY" && \
	echo "export AWS_SECRET_ACCESS_KEY=$$SECRET_KEY" && \
	echo "export PICKUP_PUTDOWN_STORAGE_BUCKET_URI=$$BUCKET_URI" && \
	echo "export PICKUP_PUTDOWN_STORAGE_REGION=$$REGION" && \
	if [ -n "$$ENDPOINT" ]; then echo "export PICKUP_PUTDOWN_STORAGE_ENDPOINT_URL=$$ENDPOINT"; fi && \
	if [ "$$ANON" = "y" ] || [ "$$ANON" = "Y" ]; then echo "export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=true"; fi
