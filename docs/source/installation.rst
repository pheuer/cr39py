Installation
============

.. _installation:

If you plan on using released versions of ``cr39py`` and do not plan to make any changes to the source code, follow the regular installation instructions.
If you want access to the development version of ``cr39py`` (e.g. the ``main`` branch  on the `GitHub repository <https://github.com/pheuer/cr39py>`_), or if you plan to make edits to the source code, follow the developer
installation instructions.


Regular Installation
--------------------

The most recent release of ``cr39py`` can be installed from PyPI using pip.

.. code-block:: console

   pip install cr39py



Developer (editable) Installation
---------------------------------

First, create a local clone of the `GitHub repository <https://github.com/pheuer/cr39py>`_. This will require a local install of git.

.. code-block:: console

   git clone https://github.com/pheuer/cr39py


Then, navigate to the folder and run the following command

.. code-block:: console

   pip install -e .

If you plan to contribute to the code, you should also install the development dependencies.

.. code-block:: console

   pip install -e .[docs,tests]
