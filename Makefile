PYTHON?=python

all: flake8 pylint

test:
	$(PYTHON) -m pytest -rswx -v -s --durations=25

black:
	$(PYTHON) -m black pyshocks tests examples

flake8:
	$(PYTHON) -m flake8 pyshocks examples tests docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint pyshocks examples/*.py tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

mypy:
	$(PYTHON) -m mypy --show-error-codes pyshocks tests examples
	@echo -e "\e[1;32mmypy clean!\e[0m"

reuse:
	@reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"

pip-install:
	$(PYTHON) -m pip install --upgrade pip numpy
	$(PYTHON) -m pip install -e '.[dev,pyweno]'

ctags:
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python

clean:
	find . -name "*.png" -exec rm -rf {} +

.PHONY: all black flake8 pylint clean ctags pip-install
