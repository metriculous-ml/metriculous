<p align="center">
    <a href="https://mybinder.org/v2/gh/metriculous-ml/metriculous/master?filepath=notebooks">
        <img alt="Launch Binder" src="https://mybinder.org/badge_logo.svg">
    </a>
    <a href="https://github.com/ambv/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
</p>

# __`metriculous`__
Unstable python library with utilities to measure, visualize and compare statistical properties of machine learning models. Breaking improvements to be expected.


# Installation
```console
$ pip install metriculous
```

Or, for the latest unreleased version:
```console
$ pip install git+https://github.com/metriculous-ml/metriculous.git
```

Or, to avoid getting surprised by breaking changes:
```console
$ pip install git+https://github.com/metriculous-ml/metriculous.git@YourFavoriteCommit
```


# Usage

### Comparing Regression Models  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/metriculous-ml/metriculous/master?filepath=notebooks%2Fquickstart_regression.py)

```python
import numpy as np

# Mock the ground truth, a one-dimensional array of floats
ground_truth = np.random.random(300)

# Mock the output of a few models
perfect_model = ground_truth
noisy_model = ground_truth + 0.1 * np.random.randn(*ground_truth.shape)
random_model = np.random.randn(*ground_truth.shape)
zero_model = np.zeros_like(ground_truth)

import metriculous

metriculous.compare_regressors(
    ground_truth=ground_truth,
    model_predictions=[perfect_model, noisy_model, random_model, zero_model],
    model_names=["Perfect Model", "Noisy Model", "Random Model", "Zero Model"],
).save_html("comparison.html").display()
```

This will save an HTML file with common regression metrics and charts, and if you are working in a [Jupyter notebook](https://github.com/jupyter/notebook) will display the output right in front of you:


![Screenshot of Metriculous Regression Metrics](./imgs/metriculous_regression_screen_shot_table.png)
![Screenshot of Metriculous Regression Figures](./imgs/metriculous_regression_screen_shot_figures.png)

### Comparing Classification Models [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/metriculous-ml/metriculous/master?filepath=notebooks%2Fquickstart_classification.py)
For an example that evaluates and compares classifiers, please refer to the [quickstart notebook for classification](https://mybinder.org/v2/gh/metriculous-ml/metriculous/master?filepath=notebooks%2Fquickstart_classification.py).


# Development

### Poetry
This project uses [poetry](https://poetry.eustace.io/) to manage
dependencies. Please make sure it is installed for the required python version. Then install the dependencies with `poetry install`.

### Makefile
A Makefile is used to automate common development workflows. Type `make` or `make help` to see a list of available commands. Before commiting changes it is recommended to run `make format check test`.
