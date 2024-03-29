
install:
	pip install -e .[dev]
	pip install pre-commit
	pre-commit install

book:
	jb build docs

serve:
	cd docs/_build/html/ && python -m http.server 0

lint:
	flake8

test:
	pytest
