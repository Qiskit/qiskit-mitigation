.PHONY: clean clean-all

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage .coverage.* coverage.xml htmlcov \
		build dist *.egg-info docs/_build docs/stubs docs/jupyter_execute
	find . -name .tox -prune -o -type d \( -name __pycache__ -o -name .ipynb_checkpoints \) -prune -exec rm -rf {} +

clean-all: clean
	rm -rf .tox
