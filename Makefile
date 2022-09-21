PIP := $(shell command -v pip3 2> /dev/null || command which pip 2> /dev/null)
PYTHON := $(shell command -v python3 2> /dev/null || command which python 2> /dev/null)

.PHONY: install dev-install install_conda dev-install_conda tests doc docupdate

pipcheck:
ifndef PIP
	$(error "Ensure pip or pip3 are in your PATH")
endif
	@echo Using pip: $(PIP)

pythoncheck:
ifndef PYTHON
	$(error "Ensure python or python3 are in your PATH")
endif
	@echo Using python: $(PYTHON)

install:
	make pipcheck
	$(PIP) install -r requirements.txt && $(PIP) install .

dev-install:
	make pipcheck
	$(PIP) install -r requirements-dev.txt && $(PIP) install -e .

install_conda:
	conda env create -f environment.yml && conda activate pymarchenko && pip install .

dev-install_conda:
	conda env create -f environment-dev.yml && conda activate pymarchenko && pip install -e .

tests:
	make pythoncheck
	$(PYTHON) setup.py test

doc:
	cd docssrc  && rm -rf source/api/generated && rm -rf source/gallery &&\
	rm -rf source/tutorials && rm -rf source/examples &&\
	rm -rf build && make html && cd ..

docupdate:
	cd docssrc && make html && cd ..

docgithub:
	cd docssrc && make github && cd ..

docpush:
	git checkout gh-pages && git merge main && cd docssrc && make github &&\
	cd ../docs && git add . && git commit -m "Updated documentation" &&\
	git push origin gh-pages && git checkout main

