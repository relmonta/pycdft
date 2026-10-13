# Convenience targets for development.
# The package core is pure numpy; the cross tests additionally need ibicus.

PYTHON ?= python

.PHONY: test test-core test-cross lint install editable clean examples

test:
	$(PYTHON) -m pytest

test-core:
	$(PYTHON) -m pytest -m "not cross"

test-cross:
	$(PYTHON) -m pytest -m cross -s

lint:
	$(PYTHON) -m pyflakes pycdft tests examples || true

install:
	$(PYTHON) -m pip install .

editable:
	$(PYTHON) -m pip install -e .[test]

examples:
	$(PYTHON) examples/example_r_gaussian.py
	$(PYTHON) examples/example_skewed.py
	$(PYTHON) examples/example_grid.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} \;
