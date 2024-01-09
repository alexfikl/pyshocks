PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

format: black	## Run all formatting scripts
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	$(PYTHON) -m isort src tests examples docs drivers
.PHONY: format

fmt: format
.PHONY: fmt

black:			## Run black over the source code
	$(PYTHON) -m black src tests examples docs drivers
.PHONY: black

lint: ruff mypy doc8 codespell reuse manifest	## Run linting checks
.PHONY: lint

ruff:			## Run ruff checks over the source code
	ruff check src tests examples docs drivers
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy src tests examples drivers
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

doc8:			## Run doc8 checks over the source code
	$(PYTHON) -m doc8 src docs
	@echo -e "\e[1;32mdoc8 clean!\e[0m"
.PHONY: doc8

codespell:		## Run codespell checks over the documentation
	@codespell --summary \
		--skip _build --skip src/*.egg-info \
		--uri-ignore-words-list '*' \
		--ignore-words .codespell-ignore \
		src tests examples docs drivers README.rst
	@echo -e "\e[1;32mcodespell clean!\e[0m"
.PHONY: codespell

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

manifest:		## Update MANIFEST.in file
	$(PYTHON) -m check_manifest
	@echo -e "\e[1;32mMANIFEST.in is up to date!\e[0m"
.PHONY: manifest

# }}}

# {{{ testing

REQUIREMENTS=\
	requirements-build.txt \
	requirements-dev.txt \
	requirements.txt

requirements-build.txt: requirements-build.in
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --strip-extras --upgrade \
		-o $@ $<

requirements-dev.txt: pyproject.toml
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --strip-extras --upgrade \
		--extra dev --extra vis --extra pyweno \
		-o $@ $<

requirements.txt: pyproject.toml
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --strip-extras --upgrade \
		-o $@ $<

pin: $(REQUIREMENTS)	## Pin dependencies versions to requirements.txt
.PHONY: pin

pip-install:			## Install pinned depdencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -r requirements-build.txt
	$(PYTHON) -m pip install -r requirements-dev.txt -e .
.PHONY: pip-install

docs:			## Generate HTML documentation (at docs/_build/html/index.html)
	(cd docs; rm -rf _build; make html SPHINXOPTS="-W --keep-going -n")
.PHONY: docs

test:					## Run pytest tests
	$(PYTHON) -m pytest -rswx --durations=25 -v -s
.PHONY: test

run-examples:	## Run examples with default options
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}" || exit $$?; \
		sleep 1; \
	done
.PHONY: run-examples

run-drivers:	## Run drivers with default options
	@for ex in $$(find drivers -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}" || exit $$?; \
		sleep 1; \
	done
.PHONY: run-drivers

# }}}

view:			## Open documentation (Linux only)
	xdg-open docs/_build/html/index.html
.PHONY: view

ctags:			## Regenerate ctags
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python
.PHONY: ctags
