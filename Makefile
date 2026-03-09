# ================================================
# HyPhi — Makefile
# ================================================
# Provides standard entry points for reproducibility.

.PHONY: install test run-simulations clean

# Install all dependencies
install:
	pip install -r requirements.txt

# Run unit tests
test:
	pytest tests/ -v

# Run full pipeline (commented out — uncomment as needed)
run-simulations:
	python main.py

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
