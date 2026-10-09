.PHONY: help lint format check fix run sync

# Support passing positional script path: make run path/to/script.py
ifeq (run,$(firstword $(MAKECMDGOALS)))
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

help:
	@echo "Available commands:"
	@echo "  make sync          - Synchronize project virtualenv with uv"
	@echo "  make lint          - Run ruff linter"
	@echo "  make format        - Format code with ruff"
	@echo "  make check         - Check linting and format status without changes"
	@echo "  make fix           - Auto-fix linting issues and format code"
	@echo "  make run <path>    - Run a Python script under Wayland"

sync:
	uv sync

run:
	@if [ -z "$(RUN_ARGS)" ]; then \
		echo "Usage: make run <path/to/script.py>"; \
		exit 1; \
	fi
	QT_QPA_PLATFORM=xcb uv run python $(RUN_ARGS)

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .

fix:
	uv run ruff check --fix .
	uv run ruff format .
