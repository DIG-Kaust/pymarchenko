PyMarchenko
===========
This Python library provides a bag of Marchenko algorithms implemented on top of `PyLops <https://pylops.readthedocs.io>`_.

Whilst a basic implementation of the `Marchenko <https://pylops.readthedocs.io/en/latest/api/generated/pylops.waveeqprocessing.Marchenko.html#pylops.waveeqprocessing.Marchenko>`_
algorithm is implemented direcly in PyLops, a number of variants have been developed over the years. This library aims to collect
all of them in the same place and give access to them with the same API to ease switching between them and prototyping of new
algorithms.

Currently we provide the following implementations:

- Marchenko redatuming via Neumann iterative substitution (Wapenaar et al., 2014)
- Marchenko redatuming via inversion (van der Neut et al., 2017)
- Rayleigh-Marchenko redatuming (Ravasi, 2017)
- Internal multiple elimination via Marchenko equations (Zhang et al., 2019)
- Marchenko redatuming with irregular sources (Haindl et al., 2021)

Alongside with the core algorithms, the following auxiliary tools are also provided:

- Target-oriented receiver redatuming via MDD
- Marchenko imaging
- Angle gather computation (de Bruin, Wapenaar, and Berkhout, 1990)


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started:

   installation.rst
   tutorials/index.rst

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference documentation:

   api/index.rst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting involved:

   Contributing <contributing.rst>
   Credits <credits.rst>

