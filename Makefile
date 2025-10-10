# Makefile for DSPy evaluation framework
# This file provides convenient commands for development and deployment

.PHONY: help install dev-install test lint typecheck format clean run-api run-cli docs

# Default target
help:
	@echo "DSPy Evaluation Framework"
	@echo "========================"
	@echo ""
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  dev-install   Install development dependencies"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  typecheck     Run type checking"
	@echo "  format        Format code"
	@echo "  clean         Clean build artifacts"
	@echo "  run-api       Run API server"
	@echo "  run-cli       Run CLI evaluation"
	@echo "  docs          Generate documentation"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Code quality
lint:
	ruff check src/ apps/ tests/
	ruff check src/ apps/ tests/ --fix

typecheck:
	mypy src/ apps/

format:
	black src/ apps/ tests/
	isort src/ apps/ tests/

# Cleaning
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

# Running
run-api:
	cd apps/api && python main.py

run-cli:
	cd apps/cli && python eval.py --help

run-sharktank:
	cd apps/cli && python eval.py --project sharktank --model gpt-4o-mini --num-samples 10

run-aneeta:
	cd apps/cli && python eval.py --project aneeta --model gemma-2b --num-samples 10

# Development
dev-setup: dev-install
	@echo "Development environment setup complete"

dev-reset: clean
	@echo "Development environment reset"

# Evaluation commands
eval-sharktank:
	cd apps/cli && python eval.py --project sharktank --model gpt-4o-mini --num-samples 100

eval-aneeta:
	cd apps/cli && python eval.py --project aneeta --model gemma-2b --num-samples 100

compare-runs:
	cd apps/cli && python compare.py --list

generate-report:
	cd apps/cli && python report.py --project sharktank --format html --output report.html

# Documentation
docs:
	mkdir -p docs
	cd docs && python -m pydoc -w src.core.config
	cd docs && python -m pydoc -w src.core.logging
	cd docs && python -m pydoc -w src.core.artifacts
	@echo "Documentation generated in docs/"

# Database operations
init-db:
	python scripts/init_database.py

migrate-db:
	python scripts/migrate_database.py

# Dataset operations
prepare-datasets:
	python scripts/prepare_datasets.py

validate-datasets:
	python scripts/validate_datasets.py

# Monitoring
health-check:
	curl -f http://localhost:8000/health || echo "API not running"

status:
	@echo "Service Status:"
	@echo "==============="
	@curl -s http://localhost:8000/status | python -m json.tool || echo "API not running"

# Backup and restore
backup:
	mkdir -p backups
	tar -czf backups/experiments-$(shell date +%Y%m%d-%H%M%S).tar.gz experiments/
	tar -czf backups/datasets-$(shell date +%Y%m%d-%H%M%S).tar.gz datasets/
	@echo "Backup created in backups/"

restore:
	@echo "Available backups:"
	@ls -la backups/
	@echo "Usage: make restore-backup BACKUP_FILE=backups/experiments-YYYYMMDD-HHMMSS.tar.gz"

restore-backup:
	tar -xzf $(BACKUP_FILE)
	@echo "Backup restored from $(BACKUP_FILE)"

# Security
security-scan:
	safety check
	bandit -r src/ apps/

# Performance
profile:
	python scripts/profile_performance.py

benchmark:
	python scripts/run_benchmarks.py

# Release
release:
	python scripts/release.py

# CI/CD
ci-test:
	pytest tests/ --cov=src --cov-report=xml --junitxml=test-results.xml

ci-lint:
	ruff check src/ apps/ tests/
	mypy src/ apps/

# Environment
env-check:
	@echo "Environment Check:"
	@echo "=================="
	@python --version
	@pip --version
	@echo "Python packages:"
	@pip list | grep -E "(dspy|fastapi|pydantic|numpy|pandas)"

# Quick start
quickstart: install
	@echo "Quick start complete!"
	@echo "Run 'make run-api' to start the API server"
	@echo "Run 'make eval-sharktank' to run a sample evaluation"
