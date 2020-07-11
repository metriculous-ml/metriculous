"""
This module provides various default Evaluator implementations that are useful for the
most common machine learning problems, such as classification and regression.
"""
from metriculous.evaluators._classification_evaluator import ClassificationEvaluator
from metriculous.evaluators._regression_evaluator import RegressionEvaluator
from metriculous.evaluators._segmentation_evaluator import SegmentationEvaluator

__all__ = ["ClassificationEvaluator", "RegressionEvaluator", "SegmentationEvaluator"]
