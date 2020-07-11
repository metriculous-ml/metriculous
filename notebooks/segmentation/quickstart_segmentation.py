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
# # Quickstart for Image Segmentation
#
# This notebook is intended to be an introduction to using the `SegmentationEvaluator` in `metriculous` for visualising and comparing the results of models for image segmentation tasks. This notebook builds on the concepts introduced in the `quickstart.ipynb`.

# %%
# # %load_ext autoreload
# # %autoreload 2

import numpy as np

np.random.seed(42)

# %% [markdown]
# # `SegmentationEvaluator`
#
# For this usage example we will use randomly generated images and masks and generate comparisons
# between three models

# %% [markdown]
# #### Define dataset properties

# %%
number_of_images = 5
image_height = 128
image_width = 128
data_shape = (number_of_images, image_height, image_width)
class_names = ("dog", "tree", "cat")

# %% [markdown]
# #### Generate ground truth data
#
# We assume that the labels are from `0` to `num_classes - 1` and they are in the order as mentioned
# in the `class_names` list. For illustration, in this example, the labelling will be as follows
#
# | Label  |  Class |
# |---|---|
# | 0 | dog  |
# | 1 |  tree |
# | 2 |  cat |

# %%
ground_truth = np.random.choice([0, 0, 1, 1, 2], size=data_shape)
ground_truth.shape

# %% [markdown]
# #### Generate model predictions
#
# For the purpose of this demonstration, we will generate the predictions of the three models
# randomly, however in a real use case, the predictions will come from the models that you want to
# compare and will have the **same** shape as `ground_truth`

# %%
num_models = 3
models = []

for i in range(num_models):
    models.append(
        {
            "name": f"Model {i + 1}",
            "predictions": np.random.choice([0, 1, 1, 2, 2, 2], size=data_shape),
        }
    )

# %% [markdown]
# #### Compare models

# %%
import metriculous

metriculous.compare(
    ground_truth=ground_truth,
    model_predictions=[model["predictions"] for model in models],
    model_names=[model["name"] for model in models],
    evaluator=metriculous.evaluators.SegmentationEvaluator(
        num_classes=len(class_names),
        class_names=class_names,
        class_weights=[0.11, 0.54, 0.35],
    ),
).display()

# %%
