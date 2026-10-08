.PHONY: check lint typecheck test
check: lint typecheck test
lint:
	ruff check .
typecheck:
	mypy src
test:
	pytest
