# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart for Regression

# %% [markdown]
# This notebook demonstrates the usage of the `RegressionEvaluator` to evaluate and compare regression models.
#
# Here we use random numbers to mock a ground truth array of floats and three models with varying degrees of errors:

# %%
import numpy as np

ground_truth = np.random.random(300)

# Mock the output of a perfect model
perfect_model = ground_truth

# Mock the output of a model that predicts the ground truth plus some Gaussian noise
noisy_model = ground_truth + 0.1 * np.random.randn(*ground_truth.shape)

# Mock the output of a model that produces random predictions
random_model = np.random.randn(*ground_truth.shape)

# Mock the output of a model that always predicts zero
zero_model = np.zeros_like(ground_truth)

# %%
import metriculous

metriculous.compare_regressors(
    ground_truth=ground_truth,
    model_predictions=[perfect_model, noisy_model, random_model, zero_model],
    model_names=["Perfect Model", "Noisy Model", "Random Model", "Zero Model"],
).display()

# %%
