# kbmod_ml

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/kbmod_ml?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/kbmod_ml/)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/dirac-institute/kbmod_ml/smoke-test.yml)](https://github.com/dirac-institute/kbmod_ml/actions/workflows/smoke-test.yml)
[![Codecov](https://codecov.io/gh/dirac-institute/kbmod_ml/branch/main/graph/badge.svg)](https://codecov.io/gh/dirac-institute/kbmod_ml)
[![Read The Docs](https://img.shields.io/readthedocs/kbmod-ml)](https://kbmod-ml.readthedocs.io/)
[![Benchmarks](https://img.shields.io/github/actions/workflow/status/dirac-institute/kbmod_ml/asv-main.yml?label=benchmarks)](https://dirac-institute.github.io/kbmod_ml/)

This project depends on ``fibad`` which is not yet available from PyPI. Please install it from source:
```
>> git clone https://github.com/lincc-frameworks/fibad.git
# activate the environment used for kbmod-ml
>> cd fibad
>> pip install -e .
```

## Dev Guide - Getting Started

Before installing any dependencies or writing code, it's a great idea to create a
virtual environment. LINCC-Frameworks engineers primarily use `conda` to manage virtual
environments. If you have conda installed locally, you can run the following to
create and activate a new environment.

```
>> conda create env -n <env_name> python=3.10
>> conda activate <env_name>
```

Once you have created a new environment, you can install this project for local
development using the following commands:

```
>> pip install -e .'[dev]'
>> pre-commit install
>> conda install pandoc
```

Notes:
1. The single quotes around `'[dev]'` may not be required for your operating system.
2. `pre-commit install` will initialize pre-commit for this local repository, so
   that a set of tests will be run prior to completing a local commit. For more
   information, see the Python Project Template documentation on 
   [pre-commit](https://lincc-ppt.readthedocs.io/en/latest/practices/precommit.html)
3. Install `pandoc` allows you to verify that automatic rendering of Jupyter notebooks
   into documentation for ReadTheDocs works as expected. For more information, see
   the Python Project Template documentation on
   [Sphinx and Python Notebooks](https://lincc-ppt.readthedocs.io/en/latest/practices/sphinx.html#python-notebooks)
