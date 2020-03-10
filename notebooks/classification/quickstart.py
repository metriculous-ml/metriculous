# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart
#
# This notebook intends to be a hands-on introduction that demonstrates the most important features of the `metriculous` library and explains core concepts.

# %%
# # %load_ext autoreload
# # %autoreload 2

import numpy as np

# %% [markdown]
# ## `ClassificationEvaluator`
# Let's start with a demonstration how `metriculous` can be used to evaluate and compare a set of machine learning models.
# We will train and evaluate a small set of classifiers on the Iris dataset, which is included in Scikit-Learn.
# The Iris dataset contains 150 flowers, each belonging to one of three classes: _setosa_, _versicolor_, _virginica_.
#
# To demonstrate `ClassificationEvaluator`,
# For this example we are going to load the data, then train a number of machine learning models and compare them with the ClassificationEvaluartor included in `metriculous`.

# %% [markdown]
# #### Load data

# %%
from sklearn.datasets import load_iris

iris = load_iris()
iris.keys()


# %%
iris.data.shape

# %%
iris.target.shape

# %%
list(iris.target_names)

# %% [markdown]
# #### Train models

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

train_indices, test_indices = train_test_split(
    np.arange(len(iris.data)), test_size=0.7, random_state=42
)

models = [
    (
        "LogisticRegression",
        LogisticRegression(multi_class="auto", solver="lbfgs", random_state=42),
    ),
    ("DecisionTree", DecisionTreeClassifier(random_state=42)),
    ("Dummy", DummyClassifier(strategy="stratified", random_state=42)),
    ("RandomForest", RandomForestClassifier(n_estimators=100, random_state=42)),
]

for name, model in models:
    model.fit(iris.data[train_indices], iris.target[train_indices])

# %% [markdown]
# #### Compare models
# `metriculous` provides a `Comparator` class that serves to evaluate a sequence of prediction objects against a known ground truth, and to compare them. A `Comparator` needs to be initialized with an `Evaluator` object that computes the actual performance metrics and creates charts for each of the prediction objects. A default `Evaluator` implementation named `ClassificationEvaluator` is included in `metriculous` and it aims to satisfy the most common requirements for classification problems.
#
# Let's use the two components to evaluate and compare our Iris classifiers:

# %%
import metriculous

test_targets_one_hot = np.eye(len(iris.target_names))[iris.target[test_indices]]

metriculous.Comparator(
    metriculous.evaluators.ClassificationEvaluator(
        # Note: All initialization parameters are optional.
        class_names=list(iris.target_names),
        top_n_accuracies=[1, 2, 3],
        filter_quantities=lambda quantity_name: quantity_name
        != "Average Precision setosa vs Rest",
        class_label_rotation_x=np.pi / 4,
        class_label_rotation_y=np.pi / 4,
    ),
).compare(
    ground_truth=test_targets_one_hot,
    model_predictions=[
        model.predict_proba(iris.data[test_indices]) for name, model in models
    ],
    model_names=[name for name, model in models],
    # sample_weights=np.array([0.5, 2.0, 1.0])[iris.target[test_indices]],
).display()

# %% [markdown]
# ## Concepts & Components
#
# The comparison we just saw is based on various building blocks that `metriculous` exposes to the user for customizability. Let's go through them one by one, starting with the most simple ones.
#
# ### `Quantity`
# A `Quantity` is a simple data container designed to hold the result of a measurement and some additional information. A few examples:

# %%
q1 = metriculous.Quantity(name="Cross-entropy", value=0.731, higher_is_better=False)

q1

# %%
q2 = metriculous.Quantity(
    name="Accuracy",
    value=0.93,
    higher_is_better=True,
    description="Fraction of correctly classified datapoints",
)

q2

# %%
q3 = metriculous.Quantity(
    name="Fraction of cat predictions",
    value=0.47,
    higher_is_better=None,
    description="Fraction of datapoints that were classified as class 'cat'",
)

q3

# %% [markdown]
# ### `Evaluation`
#
# An `Evaluation` consists of a model name, a list of `Quantity`s, and a list of callables that
# generate [Bokeh](https://bokeh.pydata.org/en/latest/) figures.
# Optionally, you can specify a primary metric by passing the name of one of the quanitities.
# This is to indicate which quantity should be used for model selection.

# %%
from bokeh.plotting import figure


def make_figure(title):
    p = figure(title=title)
    p.line([0, 1, 2, 3], np.random.random(size=4), line_width=2)
    return p


evaluation = metriculous.Evaluation(
    model_name="MyModel",
    quantities=[q1, q2, q3],
    lazy_figures=[lambda: make_figure("Interesting Chart for MyModel")],
    primary_metric="Accuracy",
)

evaluation

# %% [markdown]
# ### `Evaluator`
# An `Evaluator` is an interface. Implementations are expected to implement the method `evaluate` which has to return an `Evaluation`. An `Evaluator` has the purpose to compare a model prediction to the ground truth, compute various `Quantity`s and `Figure`s and return them as part of an `Evaluation` object.
#
# Let's take a look at the code:

# %%
import inspect

print(inspect.getsource(metriculous.Evaluator))

# %% [markdown]
# We have already seen an `Evaluator` implementation that is shipped with `metriculous`: `ClassificationEvaluator`, which we used above to evaluate a list of Iris classifiers.
# As a reminder, `ClassificationEvaluator` is a default implementation that aims to satisfy the most common requirements for classification problems.
# More default implementations, such as `RegressionEvaluator`, will most likely be added to future versions of the libary.
#
# Even though those default `Evaluator`s can be customized to some degree by passing settings to the constructor, you will probably run into a project were you want to implement your own project-specific `Evaluator`. Reasons might include
# * you want to measure quantities or create figures that are not included in the default implementations, and it wouldn't make sense to add them to the libary
# * you might want to pass in entirely different data structures, for example if your project is neither a classification problem nor a regression
#
# Looking into the implementation of `metriculous.evaluators.ClassificationEvaluator` can be a good starting point in case you wanto to implement your own `Evaluator`.

# %% [markdown]
# ### `Comparison`
# A `Comparison` consists of a list of `Evaluation`s. It serves to compare a collection of models. By calling the `display` method in a Jupyter notebook you can display a table showing the `Quantity`s for all the models side by side, as well as the `Figure`s contained in the `Evaluation`s.
#
# For a quick demonstration let's compare the `evaluation` defined above to a another `Evaluation`.

# %%
from dataclasses import replace

evaluation_2 = metriculous.Evaluation(
    model_name="MyModel_2",
    quantities=[
        replace(q1, value=0.71),
        replace(q2, value=0.31),
        replace(q3, value=0.13),
    ],
    lazy_figures=[lambda: make_figure("Interesting Chart for MyModel_2")],
    primary_metric="Accuracy",
)

comparison = metriculous.Comparison([evaluation, evaluation_2])
comparison

# %%
comparison.display()

# %% [markdown]
# ### `Comparator`
# Last but not least there is the `Comparator` class. It's a convenience class that ties all previous building blocks together. It get initialized with an `Evaluator1` (such as `ClassificationEvaluator` as in the example above), and can then be used to make a `Comparison` – which, in turn, can be displayed with a `display()` call.
#
# Note that the `compare` method has a very similar signature to `Evaluator.evaluate`. The important difference is that `Evaluator.evaluate` receives just a single prediction object, whereas `Comparator.compare` receives a sequence of prediction objects – with each object coming from one of the models that you want to compare.

# %%
print(inspect.getsource(metriculous.Comparator))
