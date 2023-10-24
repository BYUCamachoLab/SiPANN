
install:
	pip install -e .[dev]
	pip install pre-commit
	pre-commit install

book:
	cd docs && make html

lint:
	flake8

test:
	pytest
