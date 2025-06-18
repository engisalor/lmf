### UV

.PHONY: add-deps upgrade-deps add-dev precommit-update format reset-env environments

add-deps:
	uv add click langchain-community langchain_huggingface langchain-ollama ollama sentence-transformers

upgrade-deps:
	uv add --upgrade click langchain-community langchain_huggingface langchain-ollama ollama sentence-transformers

add-dev:
	uv add --dev pytest pre-commit ruff
	uv add sgex --optional extra
	uv run -- pre-commit install
	uv run -- pre-commit autoupdate

precommit-update:
	uv run -- pre-commit autoupdate

format:
	ruff check --select I --fix
	ruff format

reset-env:
	deactivate; \
	rm -rf .venv && \
	uv venv && \
	source .venv/bin/activate && \
	uv sync

environments:
	uv python install 3.12 3.11 3.10
	uv venv .venv312 --python 3.12
	uv venv .venv311 --python 3.11
	uv venv .venv310 --python 3.10

### package

.PHONY: h hq hp

h:
	clilm --help

hq:
	clilm query --help

hp:
	clilm prepare --help

### tests

.PHONY: test test-versions test-all

test:
	uv run -- pytest

define test_py_version
	echo ... running tests with py $(1) @ $(2)
	source $(2)/bin/activate && \
	uv run --python $(1) --active pytest ; \
	source .venv/bin/activate ; \
	echo ... reactivating py $(python --version)
endef

test-versions:
	@echo ... running test suite against older python versions
	$(call test_py_version, 3.12, .venv312)
	$(call test_py_version, 3.11, .venv311)

test-all: test test-versions
	@echo ... test suite ran against all python versions
