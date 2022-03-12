Installation
============

Common practice in the community is to set up a virtual environment
```
python -m venv --system-site-packages /path/to/env/pyshocks
```
Activate the environment with
```
source /path/to/env/pyshocks/bin/activate
```
Finally, just install the package with
```
python -m pip install -e .[dev]
```
which should download all the dependencies. See the
[official documentation](https://docs.python.org/3/tutorial/venv.html)
for more details.

Conda
=====

TODO for anyone using [conda](https://github.com/conda-forge/miniforge).

Documentation
=============

Documentation can be generated using [Sphinx](https://github.com/sphinx-doc/sphinx).
For example, to generate a nice HTML-based variant go
```
cd docs && make html
firefox _build/html/index.html
```
Sphinx ca can also generate LaTeX documentation with
```
make latex && cd _build/latex && latexrun pyshocks.tex
xdg-open pyshocks.pdf
```
