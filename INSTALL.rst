Installation
============

Common practice in the community is to set up a virtual environment

.. code:: bash

    python -m venv --system-site-packages /path/to/env/pyshocks

Activate the environment with

.. code:: bash

    source /path/to/env/pyshocks/bin/activate

Finally, just install the package with

.. code:: bash

    python -m pip install -e .[dev]

which should download all the dependencies. See the
`official documentation <https://docs.python.org/3/tutorial/venv.html>`__
for more details.

Conda
=====

TODO for anyone using `conda <https://github.com/conda-forge/miniforge>`__.
