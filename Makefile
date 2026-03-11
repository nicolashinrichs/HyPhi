.PHONY: install test format build clean pipeline

install:
	pip install -e .[dev]

test:
	pytest tests/ -v

format:
	black src/ tests/
	flake8 src/ tests/

build:
	python -m build

clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

pipeline:
	@echo "Executing HyPhi E2E Pipeline..."
	python main.py
