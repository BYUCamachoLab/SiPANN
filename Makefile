
install:
	pip install -e .
	pip install pre-commit
	pre-commit install

lint:
	flake8

test:
	pytest
