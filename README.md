# PyMarchenko

This Python library provides a bag of Marchenko algorithms implemented on top of [PyLops](https://pylops.readthedocs.io).

Whilst a basic implementation of the [Marchenko](https://pylops.readthedocs.io/en/latest/api/generated/pylops.waveeqprocessing.Marchenko.html#pylops.waveeqprocessing.Marchenko)
algorithm is available directly in PyLops, a number of variants have been developed over the years. This library aims at collecting
all of them in the same place and give access to them with a unique consistent API to ease switching between them and prototyping new
algorithms.

## Objective
Currently we provide the following implementations:

- Marchenko redatuming via Neumann iterative substitution (Wapenaar et al., 2014)
- Marchenko redatuming via inversion (van der Neut et al., 2017)
- Rayleigh-Marchenko redatuming (Ravasi, 2017)
- Internal multiple elimination via Marchenko equations (Zhang et al., 2019)
- Marchenko redatuming with irregular sources (Haindl et al., 2021)

Alongside the core algorithms, these following auxiliary tools are also provided:

- Target-oriented receiver-side redatuming via MDD
- Marchenko imaging (combined source-side Marchenko redatuming and receiver-side MDD redatuming)
- Angle gather computation (de Bruin, Wapenaar, and Berkhout, 1990)


## Getting started

You need **Python 3.6 or greater**.

#### From PyPi

```
pip install pymarchenko
```

#### From Github

You can also directly install from the main repository (although this is not reccomended)

```
pip install git+https://git@github.com/DIG-Kaust/pymarchenko.git@main
```

## Documentation
The official documentation of PyMarchenko is available [here](https://dig-kaust.github.io/pymarchenko/).

Visit this page to get started learning about the different algorithms implemented in this library.

Moreover, if you have installed PyMarchenko using the *developer environment* you can also build the documentation locally by
typing the following command:
```
make doc
```
Once the documentation is created, you can make any change to the source code and rebuild the documentation by
simply typing
```
make docupdate
```

Our documentation is hosted on Github-Pages and created with a Github-Action triggered every time a commit is made
to the main branch.

