"""
This module provides various default Evaluator implementations that are useful for the
most common machine learning problems, such as classification and regression.
"""
from metriculous.evaluators.classification._classification_evaluator import (
    ClassificationEvaluator,
)
from metriculous.evaluators.regression._regression_evaluator import RegressionEvaluator
from metriculous.evaluators.segmentation._segmentation_evaluator import (
    SegmentationEvaluator,
)

__all__ = ["ClassificationEvaluator", "RegressionEvaluator", "SegmentationEvaluator"]
