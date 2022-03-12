PYTHON?=python

all: flake8 pylint

black:
	$(PYTHON) -m black pyshocks examples tests

ctags:
	ctags --recurse=yes \
		--tag-relative=yes \
		--exclude=.git \
		--exclude=docs \
		--python-kinds=-i \
		--language-force=python

test:
	$(PYTHON) -m pytest -rswx --durations=25

flake8:
	$(PYTHON) -m flake8 pyshocks examples tests docs
	@echo -e "\e[1;32mflake8 clean!\e[0m"

pylint:
	PYTHONWARNINGS=ignore $(PYTHON) -m pylint pyshocks examples/*.py tests/*.py
	@echo -e "\e[1;32mpylint clean!\e[0m"

clean:
	find . -name "*.png" -exec rm -rf {} +

.PHONY: all black flake8 pylint clean
