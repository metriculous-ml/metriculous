from typing import Callable, Optional, Sequence, Union

import numpy as np

from metriculous import evaluators, utilities
from metriculous._comparison import (  # noqa (Comparator deprecated)
    Comparator,
    Comparison,
    compare,
)
from metriculous._evaluation import Evaluation, Evaluator, Quantity
from metriculous.evaluators import ClassificationEvaluator, RegressionEvaluator
from metriculous.evaluators._regression_figures_bokeh import DEFAULT_N_HISTOGRAM_BINS
from metriculous.evaluators._regression_utils import Floats

__all__ = [
    "compare",
    "compare_classifiers",
    "compare_regressors",
    "Comparison",
    "Evaluator",  # TODO Consider renaming to AbstractEvaluator?
    "Evaluation",
    "Quantity",
    "utilities",
    "evaluators",
    "ClassificationEvaluator",
    "RegressionEvaluator",
]


def compare_classifiers(
    ground_truth: Union[np.ndarray, Sequence[Sequence[float]]],
    model_predictions: Sequence[Union[np.ndarray, Sequence[Sequence[float]]]],
    model_names: Optional[Sequence[str]] = None,
    sample_weights: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
    one_vs_all_quantities: bool = True,
    one_vs_all_figures: bool = False,
    top_n_accuracies: Sequence[int] = (),
    filter_quantities: Optional[Callable[[str], bool]] = None,
    filter_figures: Optional[Callable[[str], bool]] = None,
    primary_metric: Optional[str] = None,
    simulated_class_distribution: Optional[Sequence[float]] = None,
    class_label_rotation_x: str = "horizontal",
    class_label_rotation_y: str = "vertical",
) -> Comparison:
    return compare(
        evaluator=ClassificationEvaluator(
            class_names=class_names,
            one_vs_all_quantities=one_vs_all_quantities,
            one_vs_all_figures=one_vs_all_figures,
            top_n_accuracies=top_n_accuracies,
            filter_quantities=filter_quantities,
            filter_figures=filter_figures,
            primary_metric=primary_metric,
            simulated_class_distribution=simulated_class_distribution,
            class_label_rotation_x=class_label_rotation_x,
            class_label_rotation_y=class_label_rotation_y,
        ),
        ground_truth=np.asarray(ground_truth),
        model_predictions=[np.asarray(proba_dist) for proba_dist in model_predictions],
        model_names=model_names,
        sample_weights=sample_weights,
    )


def compare_regressors(
    ground_truth: Floats,
    model_predictions: Sequence[Floats],
    model_names: Optional[Sequence[str]] = None,
    filter_quantities: Optional[Callable[[str], bool]] = None,
    filter_figures: Optional[Callable[[str], bool]] = None,
    n_histogram_bins: int = DEFAULT_N_HISTOGRAM_BINS,
    primary_metric: Optional[str] = None,
) -> Comparison:
    return compare(
        evaluator=RegressionEvaluator(
            filter_quantities=filter_quantities,
            filter_figures=filter_figures,
            n_histogram_bins=n_histogram_bins,
            primary_metric=primary_metric,
        ),
        ground_truth=ground_truth,
        model_predictions=model_predictions,
        model_names=model_names,
        sample_weights=None,  # sample_weights are currently not yet supported
    )
