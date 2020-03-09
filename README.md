<p align="center">
<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Metriculous
Very unstable library containing utilities to measure and visualize statistical properties of machine learning models.

## Quickstart
For examples and a general introduction please refer to 
[the quickstart notebook](notebooks/classification/quickstart.ipynb).

## Development

### Pre-commit
Please make sure to have the pre-commit hooks installed.
Install [pre-commit](https://pre-commit.com/) and then run `pre-commit install` to register the hooks with git.

### Poetry
This project uses [poetry](https://poetry.eustace.io/) to manage
dependencies. Please make sure it is installed for the required python
version. Then install the dependencies with:

```
poetry install
```

To activate the virtual environment created by `poetry`, run

```
poetry shell
```

or execute individual commands with `poetry run`, e.g.

```
poetry run jupyter notebook
```

### Makefile
Run `make help` to see all available commands.

<!-- START makefile-doc -->
```
$ make help 
help                 Show this help message
bump                 Bump metriculous version
check                Run all static checks (like pre-commit hooks)
test                 Run all tests 
```
<!-- END makefile-doc -->
