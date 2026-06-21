SHELL := /bin/bash

PYTHON ?= python

STORAGE_ENV ?= .local/env/storage.env
STORAGE_CONFIG ?= configs/storage.yaml

TRIAGE_CONFIG ?= configs/triage.yaml
TRIAGE_OUTPUT ?= .local/triage_acceptance/output_single
TRIAGE_INPUT ?= $(shell find .local \
	-type f \
	-iname '*.mp4' \
	-not -path '*/triage_acceptance/output_*/*' \
	-not -path '*/triage_previews/*' \
	2>/dev/null | sort | head -n 1)

.PHONY: help install install-dev lint format format-check typecheck test compile clean \
	env-setup env-ingest ingest task-3

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
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

env-setup: ## Prompt for storage settings and save them to a protected local environment file
	@mkdir -p "$(dir $(STORAGE_ENV))"
	@read -rp "AWS Access Key ID: " ACCESS_KEY && \
	read -rsp "AWS Secret Access Key: " SECRET_KEY && echo && \
	read -rp "S3 Bucket URI (s3://bucket/prefix): " BUCKET_URI && \
	read -rp "S3 Region (e.g. us-east-1): " REGION && \
	read -rp "Endpoint URL (leave empty for AWS default): " ENDPOINT && \
	read -rp "Anonymous access (y/N): " ANON && \
	umask 077 && \
	{ \
		printf 'export AWS_ACCESS_KEY_ID=%q\n' "$$ACCESS_KEY"; \
		printf 'export AWS_SECRET_ACCESS_KEY=%q\n' "$$SECRET_KEY"; \
		printf 'export PICKUP_PUTDOWN_STORAGE_BUCKET_URI=%q\n' "$$BUCKET_URI"; \
		printf 'export PICKUP_PUTDOWN_STORAGE_REGION=%q\n' "$$REGION"; \
		if [ -n "$$ENDPOINT" ]; then \
			printf 'export PICKUP_PUTDOWN_STORAGE_ENDPOINT_URL=%q\n' "$$ENDPOINT"; \
		fi; \
		if [[ "$$ANON" =~ ^[yY]$$ ]]; then \
			printf 'export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=true\n'; \
		else \
			printf 'export PICKUP_PUTDOWN_STORAGE_ANONYMOUS=false\n'; \
		fi; \
	} > "$(STORAGE_ENV)" && \
	chmod 600 "$(STORAGE_ENV)" && \
	echo "Storage environment saved to $(STORAGE_ENV)"

ingest: ## Run ingestion using the saved local storage environment
	@test -f "$(STORAGE_ENV)" || \
		(echo "Missing $(STORAGE_ENV). Run 'make env-setup' first." && exit 1)
	@set -a && \
	source "$(STORAGE_ENV)" && \
	set +a && \
	pickup-putdown ingest --config "$(STORAGE_CONFIG)"

env-ingest: env-setup ## Configure storage and immediately run ingestion
	@$(MAKE) ingest

task-3: ## Run Task 3 triage against the first cached MP4 found under .local
	@test -n "$(TRIAGE_INPUT)" || \
		(echo "No cached MP4 found under .local." && \
		 echo "Run 'make ingest' first or set TRIAGE_INPUT=/path/to/video.mp4." && \
		 exit 1)
	@test -f "$(TRIAGE_INPUT)" || \
		(echo "Triage input does not exist: $(TRIAGE_INPUT)" && exit 1)
	@mkdir -p models "$(TRIAGE_OUTPUT)"
	@if [ ! -f models/person_detector.pt ]; then \
		echo "Downloading YOLO11n person detector..."; \
		$(PYTHON) -c 'from ultralytics import YOLO; YOLO("yolo11n.pt")'; \
		mv -f yolo11n.pt models/person_detector.pt; \
	fi
	@echo "Triage input: $(TRIAGE_INPUT)"
	@echo "Triage output: $(TRIAGE_OUTPUT)"
	pickup-putdown triage \
		"$(TRIAGE_INPUT)" \
		--config "$(TRIAGE_CONFIG)" \
		--output-dir "$(TRIAGE_OUTPUT)" \
		--verbose
