PYTHON?=python

all: flake8 pylint

# {{{ linting

black:
	$(PYTHON) -m black --safe --target-version py38 pyshocks tests examples

flake8:
	$(PYTHON) -m flake8 pyshocks examples tests docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint pyshocks examples/*.py tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

mypy:
	$(PYTHON) -m mypy --strict --show-error-codes pyshocks tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"

codespell:
	@codespell --summary \
		--ignore-words .codespell-ignore \
		pyshocks tests examples

reuse:
	@reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"

# }}}

# {{{ testing

pip-install:
	$(PYTHON) -m pip install --upgrade pip numpy
	$(PYTHON) -m pip install -e '.[dev,pyweno]'

docs:
	(cd docs; rm -rf _build; make html SPHINXOPTS="-W --keep-going -n")

test:
	$(PYTHON) -m pytest -rswx -v -s --durations=25

run-examples:
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done

# }}}

view:
	xdg-open docs/_build/html/index.html

ctags:
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python

.PHONY: all docs view black flake8 pylint clean ctags pip-install
