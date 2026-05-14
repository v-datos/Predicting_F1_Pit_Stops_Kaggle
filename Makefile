.PHONY: test lint format format-check typecheck check

test:
	python -m pytest tests/ -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

typecheck:
	mypy src/predicting_f1_pit_stops/ --ignore-missing-imports

check: lint format-check typecheck test
