PYTHON := 'python -X dev'

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc('Reformat all source code')]
format: isort black pyproject justfmt

[doc('Run ruff isort fixes over the source code')]
isort:
    ruff check --fix --select=I src tests examples docs drivers
    ruff check --fix --select=RUF022 src
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc('Run ruff format over the source code')]
black:
    ruff format src tests examples docs drivers
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc('Run pyproject-fmt over the configuration')]
pyproject:
    {{ PYTHON }} -m pyproject_fmt \
        --indent 4 --max-supported-python '3.13' \
        pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc('Run just --fmt over the justfile')]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc('Run all linting checks over the source code')]
lint: typos reuse ruff doc8 mypy

[doc('Run typos over the source code and documentation')]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc('Run ruff checks over the source code')]
ruff:
    ruff check src tests examples docs drivers
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc('Run doc8 checks over the documentation')]
doc8:
    {{ PYTHON }} -m doc8 src docs
    @echo -e "\e[1;32mdoc8 clean!\e[0m"

[doc('Run mypy checks over the source code')]
mypy:
    {{ PYTHON }} -m mypy src tests examples drivers
    @echo -e "\e[1;32mmypy clean!\e[0m"

# }}}
# {{{ pin

[private]
requirements_build_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o .github/requirements-build.txt .github/requirements-build.in

[private]
requirements_dev_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        --extra dev --extra vis \
        -o .github/requirements-dev.txt pyproject.toml

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt requirements_dev_txt requirements_build_txt

# }}}
# {{{ develop

[doc('Install project in editable mode')]
develop:
    @rm -rf build
    @rm -rf dist
    {{ PYTHON }} -m pip install \
        --verbose \
        --no-build-isolation \
        --editable .

[doc("Editable install using pinned dependencies from requirements-dev.txt")]
pip-install:
    {{ PYTHON }} -m pip install --verbose --requirement .github/requirements-build.txt
    {{ PYTHON }} -m pip install \
        --verbose \
        --requirement requirements-dev.txt \
        --no-build-isolation \
        --editable .
    {{ PYTHON }} -m pip install --verbose --no-build-isolation \
        'git+https://github.com/alexfikl/PyWENO.git@numpy-2.0#egg=PyWENO'

[doc("Remove various build artifacts")]
clean:
    rm -rf *.png
    rm -rf build dist
    rm -rf docs/build.sphinx

[doc("Remove various temporary files and caches")]
purge: clean
    rm -rf .ruff_cache .pytest_cache .pytest-cache .mypy_cache tags

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

[doc("Open documentation in your various browser")]
view opener="xdg-open":
    {{ opener }} docs/build.sphinx/html/index.html

# }}}
# {{{ test

[doc("Run pytest tests")]
test *PYTEST_ADDOPTS:
    {{ PYTHON }} -m pytest -rswx --durations=25 -v -s {{ PYTEST_ADDOPTS }}

[doc("Run examples with default options")]
examples:
    @for ex in $(ls examples/*.py); do \
        echo "::group::Running ${ex}"; \
        {{ PYTHON }} ${ex}; \
        sleep 1; \
        echo "::endgroup::"; \
    done

[doc("Run drivers with default options")]
drivers:
    @for ex in $(ls drivers/*.py); do \
        echo "::group::Running ${ex}"; \
        {{ PYTHON }} ${ex}; \
        sleep 1; \
        echo "::endgroup::"; \
    done

# }}}
