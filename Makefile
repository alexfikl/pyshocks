PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ linting

fmt: black		## Run all formatting scripts
	$(PYTHON) -m setup_cfg_fmt --include-version-classifiers setup.cfg
	$(PYTHON) -m pyproject_fmt --indent 4 pyproject.toml
	$(PYTHON) -m isort pyshocks tests examples docs drivers
.PHONY: fmt

black:			## Run black over the source code
	$(PYTHON) -m black --safe --target-version py38 pyshocks tests examples drivers
.PHONY: black

flake8:			## Run flake8 checks over the source code
	$(PYTHON) -m flake8 pyshocks examples drivers tests docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"
.PHONY: flake8

pylint:			## Run pylint checks over the source code
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint pyshocks examples/*.py drivers/*.py tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"
.PHONY: pylint

mypy:			## Run mypy checks over the source code
	$(PYTHON) -m mypy \
		--strict --show-error-codes \
		pyshocks tests examples drivers
	@echo -e "\e[1;32mmypy clean!\e[0m"
.PHONY: mypy

doc8:			## Run doc8 checks over the source code
	PYTHONWARNINGS=ignore $(PYTHON) -m doc8 pyshocks docs
	@echo -e "\e[1;32mdoc8 clean!\e[0m"
.PHONY: doc8

linkcheck:		## Run sphinx linkcheck checks over the documentation
	cd docs && make linkcheck SPHINXOPTS="-W --keep-going -n" || true
	@echo -e "\e[1;32mlinkcheck clean!\e[0m"
.PHONY: linkcheck

sphinxlint: linkcheck doc8		## Run sphinxlint checks over the source code
	$(PYTHON) -m sphinxlint -v \
		--ignore docs/_build --ignore README.rst \
		--enable all \
		--max-line-length=88 \
		.
	@echo -e "\e[1;32msphinxlint clean!\e[0m"
.PHONY: sphinxlint

codespell:		## Run codespell checks over the documentation
	codespell --summary \
		--ignore-words .codespell-ignore \
		pyshocks tests examples drivers README.rst
	@echo -e "\e[1;32mcodespell clean!\e[0m"
.PHONY: codespell

reuse:			## Check REUSE license compliance
	reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

# }}}

# {{{ alternative linting

pyright:		## Run pyright checks over the source code
	pyright --stats pyshocks tests examples drivers
	@echo -e "\e[1;32mpyright clean!\e[0m"
.PHONY: pyright

ruff:			## Run ruff checks over the source code
	ruff pyshocks tests examples drivers
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

pytype:			## Run pytype checks over the source code
	$(PYTHON) -m pytype \
		--strict-parameter-checks \
		--strict-primitive-comparisons \
		pyshocks tests examples drivers
	@echo -e "\e[1;32mpytype clean!\e[0m"
.PHONY: pytype

# }}}

# {{{ testing

pin:			## Pin dependencies versions to requirements.txt
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		requirements-build.in
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		--extra dev --extra pyweno \
		-o requirements.txt setup.cfg
.PHONY: pin

pip-install:	## Install pinned depdencies from requirements.txt
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -r requirements-build.txt
	$(PYTHON) -m pip install -r requirements.txt -e .
.PHONY: pip-install

docs:			## Generate HTML documentation (at docs/_build/html/index.html)
	(cd docs; rm -rf _build; make html SPHINXOPTS="-W --keep-going -n")
.PHONY: docs

test:			## Run pytest tests
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
