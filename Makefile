# ================================================
# HyPhi — Makefile
# ================================================
# Provides standard entry points for reproducibility.

.PHONY: help install test check lint typecheck format clean pipeline
.DEFAULT_GOAL := help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | awk -F ':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync --extra develop --extra notebook

test: ## Run tests with coverage
	uv run --extra develop pytest . --cov-report=html -v

check: format typecheck lint ## Run format, typecheck, and lint checks

lint: ## Lint code with ruff
	uv run --extra develop ruff check code/hyphi --fix

typecheck: ## Type-check code with ty
	uv run --extra develop ty check code/hyphi

format: ## Format code with ruff
	uv run --extra develop ruff format code/tests
	uv run --extra develop ruff format code/hyphi
	uv run --extra develop ruff format experiments/scripts

clean: ## Remove build artifacts and caches
	rm -rf code/hyphi.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


run-simulations: ## Execute HyPhi E2E simulations
	@echo "Executing HyPhi E2E Pipeline..."
	uv run python -m hyphi.main
