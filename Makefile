PYTHON?=python -X dev

all: flake8 pylint

# {{{ linting

black:
	$(PYTHON) -m black --safe --target-version py38 pyshocks tests examples drivers

flake8:
	$(PYTHON) -m flake8 pyshocks examples drivers tests docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint pyshocks examples/*.py drivers/*.py tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

mypy:
	$(PYTHON) -m mypy --strict --show-error-codes pyshocks tests examples drivers
	@echo -e "\e[1;32mmypy clean!\e[0m"

linkcheck:
	cd docs && make linkcheck SPHINXOPTS="-W --keep-going -n" || true
	@echo -e "\e[1;32mlinkcheck clean!\e[0m"

sphinxlint: linkcheck
	$(PYTHON) -m sphinxlint -v \
		--ignore docs/_build --ignore README.rst \
		--enable all \
		--max-line-length=88 \
		.
	@echo -e "\e[1;32msphinxlint clean!\e[0m"

codespell:
	codespell --summary \
		--ignore-words .codespell-ignore \
		pyshocks tests examples drivers
	@echo -e "\e[1;32mcodespell clean!\e[0m"

reuse:
	reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"

# }}}

# {{{ alternative linting

pyright:
	pyright --stats pyshocks tests examples drivers
	@echo -e "\e[1;32mpyright clean!\e[0m"

ruff:
	ruff pyshocks tests examples drivers
	@echo -e "\e[1;32mruff clean!\e[0m"

pytype:
	$(PYTHON) -m pytype \
		--strict-parameter-checks \
		--strict-primitive-comparisons \
		pyshocks tests examples drivers
	@echo -e "\e[1;32mpytype clean!\e[0m"

# }}}

# {{{ testing

pin:
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		requirements-build.in
	$(PYTHON) -m piptools compile \
		--resolver=backtracking --upgrade \
		--extra dev --extra pyweno \
		-o requirements.txt setup.cfg

pip-install:
	$(PYTHON) -m pip install --upgrade pip setuptools
	$(PYTHON) -m pip install -r requirements-build.txt
	$(PYTHON) -m pip install -r requirements.txt -e .

docs:
	(cd docs; rm -rf _build; make html SPHINXOPTS="-W --keep-going -n")

test:
	$(PYTHON) -m pytest -rswx --durations=25 -v -s

run-examples:
	@for ex in $$(find examples -name "*.py"); do \
		echo -e "\x1b[1;32m===> \x1b[97mRunning $${ex}\x1b[0m"; \
		$(PYTHON) "$${ex}"; \
		sleep 1; \
	done

run-drivers:
	@for ex in $$(find drivers -name "*.py"); do \
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
