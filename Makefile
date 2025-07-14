.PHONY: help install install-dev format lint test clean pre-commit docker-build docker-run

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"

format:  ## Format code with black and isort
	black .
	isort .

lint:  ## Run linting tools
	flake8 .
	pylint *.py

test:  ## Run tests
	pytest -v --cov=. --cov-report=html --cov-report=term

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit:  ## Install pre-commit hooks
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

docker-build:  ## Build Docker image
	docker build -t whisper-cli .

docker-run:  ## Run Docker container
	docker run --rm -it -p 8000:8000 -e MODE=web whisper-cli

docker-test:  ## Run tests in Docker
	docker run --rm -v $(PWD):/app -w /app python:3.11-slim bash -c "pip install -e .[dev] && pytest -v"

ci-check:  ## Run all CI checks locally
	make format
	make lint
	make test

dev-setup:  ## Complete development setup
	make install-dev
	make pre-commit
	@echo "Development environment ready!"
