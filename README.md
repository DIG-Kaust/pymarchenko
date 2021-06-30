# PyMarchenko

This Python library provides a bag of Marchenko algorithms implemented on top of [PyLops](https://pylops.readthedocs.io).

Whilst a basic implementation of the [Marchenko](https://pylops.readthedocs.io/en/latest/api/generated/pylops.waveeqprocessing.Marchenko.html#pylops.waveeqprocessing.Marchenko).
algorithm is implemented direcly in PyLops, a number of variants have been developed over the years. This library aims to collect
all of them in the same place and give access to them with the same API to ease switching between them and prototyping of new
algorithms.

Currently we provide the following implementations:

- Marchenko redatuming via Neumann iterative substitution (Wapenaar et al., 2014)
- Marchenko redatuming via inversion (van der Neut et al., 2017)
- Rayleigh-Marchenko redatuming (Ravasi, 2017)
- Internal multiple elimination via Marchenko equations (Zhang and Staring, 2018)
- *TO DO*: Marchenko redatuming with irregular sources (Haindl et al., 2021)

Alongside with the core algorithms, the following auxiliary tools are also provided:

- Target-oriented receiver redatuming via MDD
- Marchenko imaging
- Angle gather computation (de Bruin, Wapenaar, and Berkhout, 1990)


## Installation
Bla bla

Note that to test the algorithms in this library we need access to test data that exceeds the 100Mb limit of GitHub. 
We are therefore using ``git-lfs`` to store such files. You will need to install this utility tool prior to 
begin following these [instructions]( http://arfc.github.io/manual/guides/git-lfs). After that you will be able to clone 
the repo and the test data will be available in the same way as if they were directly committed in the repository.

For developers, to add a new dataset simply run

```
git lfs track "FILE"

git add FILE
git commit -m "Added file"
git push origin main
```