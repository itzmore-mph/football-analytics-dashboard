.PHONY: help setup install run test lint format clean demo

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## Setup virtual environment and install dependencies
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  (Linux/macOS)"
	@echo "  .venv\\Scripts\\activate     (Windows)"

install: ## Install package with dev dependencies
	pip install -U pip
	pip install -e .[dev]
	pre-commit install

run: ## Launch Streamlit dashboard
	streamlit run streamlit_app.py

test: ## Run tests with pytest
	pytest tests/ -v

lint: ## Check code quality with ruff
	ruff check src/ tests/

format: ## Format code with ruff, black, and isort
	ruff check src/ tests/ --fix
	black src/ tests/
	isort src/ tests/

clean: ## Clean up cache and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

demo: ## Run full demo pipeline (fetch, preprocess, train, evaluate)
	python -m src.cli demo
