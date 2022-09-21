.. _installation:

============
Installation
============

To install the PyMarchenko library you will need **Python 3.6 or greater**.


Dependencies
------------

Our mandatory dependencies are limited to:

* `numpy <http://www.numpy.org>`_
* `scipy <http://www.scipy.org/scipylib/index.html>`_
* `pylops <https://pylops.readthedocs.io>`_

We advise using the `Anaconda Python distribution <https://www.anaconda.com/download>`_
to ensure that these dependencies are installed via the ``Conda`` package manager. This
is not just a pure stylistic choice but comes with some *hidden* advantages, such as the linking to
``Intel MKL`` library (i.e., a highly optimized BLAS library created by Intel).


Step-by-step installation for users
-----------------------------------

Activate your Python environment, and simply type the following command in your terminal
to install the PyPi distribution:

.. code-block:: bash

   >> pip install pymarchenko

Alternatively, to access the latest source from github:

.. code-block:: bash

   >> pip install https://github.com/DIG-Kaust/pymarchenko/archive/main.zip

or just clone the repository

.. code-block:: bash

   >> git clone https://github.com/DIG-Kaust/pymarchenko.git

or download the zip file from the repository (green button in the top right corner of the
main github repo page) and install PyMarchenko from terminal using the command:

.. code-block:: bash

   >> make install


Step-by-step installation for developers
----------------------------------------

Python environment
~~~~~~~~~~~~~~~~~~

Fork and clone the repository by executing the following in your terminal:

.. code-block:: bash

   >> git clone https://github.com/your_name_here/pymarchenko.git

The first time you clone the repository run the following command:

.. code-block:: bash

   >> make dev-install

If you prefer to build a new Conda enviroment just for PyMarchenko, run the following command:

.. code-block:: bash

   >> make dev-install_conda

To ensure that everything has been setup correctly, run tests:

.. code-block:: bash

    >> make tests

Make sure no tests fail, this guarantees that the installation has been successfull.

If using Conda environment, always remember to activate the conda environment every time you open
a new *bash* shell by typing:

.. code-block:: bash

   >> conda activate pymarchenko


Documentation
~~~~~~~~~~~~~

You can also build the documentation locally by typing the following command:

.. code-block:: bash

   >> make doc

Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing

.. code-block:: bash

   >> make docupdate

Since the tutorials are too heavy to be created by documentation web-services like Readthedocs, our documentation
is hosted on Github-Pages and run locally on a separate branch. To get started create the following branch both locally
and in your remote fork:

.. code-block:: bash

   >> git checkout -b gh-pages
   >> git push -u origin gh-pages

Every time you want to update and deploy the documentation run:

.. code-block:: bash

   >> make docpush

This will automatically move to the `gh-pages` branch, build the documentation and push it in the equivalent remote branch.
You can finally make a Pull Request for your local `gh-pages` branch to the `gh-pages` in the DIG-Kaust repository,

