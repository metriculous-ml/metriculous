# -*- coding: utf-8 -*-
from setuptools import setup

package_dir = {"": "src"}

packages = ["metriculous", "metriculous.evaluators"]

package_data = {"": ["*"]}

install_requires = [
    "assertpy>=0.14.0",
    "bokeh>=1.1",
    "numpy>=1.16",
    "pandas>=0.24.0",
    "scikit-learn>=0.21.2",
]

setup_kwargs = {
    "name": "metriculous",
    "version": "0.1.0",
    "description": "Very unstable library containing utilities to measure and visualize statistical properties of machine learning models.",
    "long_description": '<p align="center">\n<a href="https://github.com/ambv/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>\n</p>\n\n# Metriculous\nVery unstable library containing utilities to measure and visualize statistical properties of machine learning models.\n\n## Quickstart\nFor examples and a general introduction please refer to [the quickstart notebook](./notebooks/quickstart.ipynb).\n\n## Development\n\n### Pre-commit\nPlease make sure to have the pre-commit hooks installed.\nInstall [pre-commit](https://pre-commit.com/) and then run `pre-commit install` to register the hooks with git.\n\n### Poetry\nThis project uses [poetry](https://poetry.eustace.io/) to manage\ndependencies. Please make sure it is installed for the required python\nversion. Then install the dependencies with:\n\n```\npoetry install\n```\n\nTo activate the virtual environment created by `poetry`, run\n\n```\npoetry shell\n```\n\nor execute individual commands with `poetry run`, e.g.\n\n```\npoetry run jupyter notebook\n```\n\n### Makefile\nRun `make help` to see all available commands.\n\n<!-- START makefile-doc -->\n```\n$ make help \nhelp                 Show this help message\nbump                 Bump metriculous version\ncheck                Run all static checks (like pre-commit hooks)\ntest                 Run all tests \n```\n<!-- END makefile-doc -->\n',
    "author": "Luminovo GmbH",
    "author_email": "pypi@luminovo.ai",
    "maintainer": "Marlon",
    "maintainer_email": "marlon@luminovo.ai",
    "url": "https://gitlab.com/luminovo/public/metriculous",
    "package_dir": package_dir,
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "python_requires": ">=3.7,<4.0",
}


setup(**setup_kwargs)
