.. _contributing:

Contributing
============

Contributions are welcome and greatly appreciated!

The best way to get in touch with the core developers and mantainers is to
open new *Issues* directly from the
`github repo <https://github.com/DIG-KAUST/pymarchenko>`_.

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/DIG-KAUST/pymarchenko.

If you are playing with the PyMarchenko library and find a bug, please report it including:

* Your operating system name and version.
* Any details about your Python environment.
* Detailed steps to reproduce the bug.

Propose New Algorithm
~~~~~~~~~~~~~~~~~~~~~

Open an issue at https://github.com/DIG-KAUST/pymarchenko/issues with tag *enhancement*.

Advices when opening an issue of this kind:

* Explain in detail how it should work.
* Keep the scope as narrow as possible, to make it easier to implement.

Add Examples or improve Documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Writing new algorithms is not the only way to get involved and contribute. Create examples with existing algorithms
as well as improving the documentation of existing algorithms is as important as making new algorithms and very much
encouraged.


Getting Started to contribute
-----------------------------

Ready to contribute?

1. Fork the `PyMarchenko` repo.

2. Clone your fork locally:

.. code-block:: bash

   >>  git clone https://github.com/your_name_here/pymarchenko.git

3. Follow the installation instructions for *developers* that you find in :ref:`installation` page.
   Ensure that you are able to *pass all the tests before moving forward*.

4. Add the main repository to the list of your remotes (this will be important to ensure you
   pull the latest changes before tyring to merge your local changes):

.. code-block:: bash

   >>  git remote add upstream https://github.com/DIG-KAUST/pymarchenko

5. Create a branch for local development:

.. code-block:: bash

   >>  git checkout -b name-of-your-branch

Now you can make your changes locally.

6. When you're done making changes, check that your code follows the guidelines for :ref:`addingoperator` and
that the both old and new tests pass successfully:

.. code-block:: bash

   >>  make tests

7. Commit your changes and push your branch to GitLab:

.. code-block:: bash

   >>  git add .
   >> git commit -m "Your detailed description of your changes."
   >> git push origin name-of-your-branch

Remember to add ``-u`` when pushing the branch for the first time.

8. Submit a pull request through the GitHub website.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include new tests for all the core routines that have been developed.
2. If the pull request adds functionality, the docs should be updated accordingly.
3. Ensure that the updated code passes all tests.