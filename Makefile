.PHONY: test lint format typecheck build clean help install docs test-cov benchmark examples all

help:
	@echo "Agentick Development Commands"
	@echo "============================="
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make lint       - Run linter (ruff check)"
	@echo "  make format     - Format code (ruff format)"
	@echo "  make typecheck  - Run type checker (mypy)"
	@echo "  make docs       - Build documentation"
	@echo "  make build      - Build distribution packages"
	@echo "  make benchmark  - Run performance benchmarks"
	@echo "  make examples   - Run all examples"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make all        - Run all checks"

install:
	uv pip install -e ".[dev]"

docs:
	mkdocs build

test-cov:
	uv run pytest tests/ --cov=agentick --cov-report=html --cov-report=term

benchmark:
	uv run pytest tests/test_performance/ -v -m benchmark

examples:
	@echo "Running examples..."
	@for script in examples/*.py; do \
		echo "Running $$script..."; \
		uv run python $$script || exit 1; \
	done

all: lint typecheck test build

test:
	uv run pytest tests/ -v --timeout=300

lint:
	uv run ruff check agentick/

format:
	uv run ruff format agentick/ tests/ examples/

typecheck:
	uv run mypy agentick/ --ignore-missing-imports

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
