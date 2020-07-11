from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from bokeh.plotting import Figure
from sklearn import metrics as sklmetrics

from .._evaluation import Evaluation, Evaluator, Quantity
from ._regression_figures_bokeh import (
    DEFAULT_N_HISTOGRAM_BINS,
    _bokeh_actual_vs_predicted_scatter_with_histograms,
    _bokeh_residual_histogram,
    _bokeh_residual_vs_predicted_scatter_with_histograms,
    _bokeh_targets_and_predictions_histograms,
)
from ._regression_utils import Floats, RegressionData


class RegressionEvaluator(Evaluator[Floats, Floats]):
    """
    Evaluator implementation for regression problems.
    """

    def __init__(
        self,
        filter_quantities: Optional[Callable[[str], bool]] = None,
        filter_figures: Optional[Callable[[str], bool]] = None,
        n_histogram_bins: int = DEFAULT_N_HISTOGRAM_BINS,
        primary_metric: Optional[str] = None,
    ):
        """
        Initializes the evaluator with the option to overwrite the default settings.

        Args:
            filter_quantities:
                Callable that receives a quantity name and returns `False` if the
                quantity should be excluded.
                Example: `filter_quantities=lambda name: "vs Rest" not in name`
            filter_figures:
                Callable that receives a figure title and returns `False` if the figure
                should be excluded.
                Example: `filter_figures=lambda name: "ROC" in name`
            n_histogram_bins:
                 Number of bins for histograms in the figures.
            primary_metric:
                Optional string to specify the most important metric that should be used
                for model selection.

        """
        self.filter_quantities = (
            (lambda name: True) if filter_quantities is None else filter_quantities
        )
        self.filter_figures = (
            (lambda name: True) if filter_figures is None else filter_figures
        )
        self.n_histogram_bins = n_histogram_bins
        self.primary_metric = primary_metric

    def evaluate(
        self,
        ground_truth: Floats,
        model_prediction: Floats,
        model_name: str,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Evaluation:
        """
        Computes Quantities and generates Figures that are useful for most
        regression problems.

        Args:
            ground_truth:
                1d float array with ground truth values.
            model_prediction:
                1d float array with values predicted by the model.
            model_name:
                Name of the model that is being evaluated.
            sample_weights:
                Sequence of floats to modify the influence of individual samples on the
                statistics that will be measured.

        Returns:
            An Evaluation object containing Quantities and Figures that are useful for
            most regression problems.

        """

        # === Preparations =============================================================
        data = RegressionData(targets=ground_truth, predictions=model_prediction)

        if sample_weights is not None:
            assert len(data.targets) == len(
                sample_weights
            ), f"{len(data.targets)} != {len(sample_weights)}"
            sample_weights = np.asarray(sample_weights)

        quantities = [
            q
            for q in _quantities(data=data, maybe_sample_weights=sample_weights)
            if self.filter_quantities(q.name)
        ]

        unfiltered_lazy_figures = _lazy_figures(
            data=data,
            model_name=model_name,
            maybe_sample_weights=sample_weights,
            n_histogram_bins=self.n_histogram_bins,
        )

        return Evaluation(
            quantities=quantities,
            lazy_figures=[
                function
                for name, function in unfiltered_lazy_figures
                if self.filter_figures(name)
            ],
            model_name=model_name,
            primary_metric=self.primary_metric,
        )


class FigureNames:
    HISTOGRAM_COMPARISON = "Histogram Comparison"
    ACTUAL_VS_PREDICTED_SCATTER = "Actual vs Predicted Scatter"
    RESIDUAL_VS_PREDICTED_SCATTER = "Residual vs Predicted Scatter"
    RESIDUAL_HISTOGRAM = "Residual Histogram"


def _lazy_figures(
    data: RegressionData,
    model_name: str,
    maybe_sample_weights: Optional[Floats],
    n_histogram_bins: int,
) -> Sequence[Tuple[str, Callable[[], Figure]]]:
    F = FigureNames

    if maybe_sample_weights is not None:
        return []

    # TODO include model names in chart titles

    lazy_figures = [
        (
            F.HISTOGRAM_COMPARISON,
            _bokeh_targets_and_predictions_histograms(
                data=data,
                title_rows=[model_name, F.HISTOGRAM_COMPARISON],
                n_bins=n_histogram_bins,
            ),
        ),
        (
            F.ACTUAL_VS_PREDICTED_SCATTER,
            _bokeh_actual_vs_predicted_scatter_with_histograms(
                data=data,
                title_rows=[model_name, F.ACTUAL_VS_PREDICTED_SCATTER],
                n_bins=n_histogram_bins,
            ),
        ),
        (
            F.RESIDUAL_VS_PREDICTED_SCATTER,
            _bokeh_residual_vs_predicted_scatter_with_histograms(
                data=data,
                title_rows=[model_name, F.RESIDUAL_VS_PREDICTED_SCATTER],
                n_bins=n_histogram_bins,
            ),
        ),
        (
            F.RESIDUAL_HISTOGRAM,
            _bokeh_residual_histogram(
                data=data, title_rows=[model_name, F.RESIDUAL_HISTOGRAM]
            ),
        ),
    ]

    return lazy_figures


def _quantities(
    data: RegressionData, maybe_sample_weights: Optional[Floats]
) -> Sequence[Quantity]:
    y_true = data.targets
    y_pred = data.predictions

    quantities = [
        # --- Scores (higher is better) ---
        Quantity(
            "R^2",
            sklmetrics.r2_score(
                y_true=y_true, y_pred=y_pred, sample_weight=maybe_sample_weights
            ),
            higher_is_better=True,
        ),
        Quantity(
            "Explained Variance",
            sklmetrics.explained_variance_score(
                y_true=y_true, y_pred=y_pred, sample_weight=maybe_sample_weights
            ),
            higher_is_better=True,
        ),
        # --- Errors (lower is better) ---
        Quantity(
            "MSE (Mean Squared Error)",
            sklmetrics.mean_squared_error(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=maybe_sample_weights,
                squared=True,
            ),
            higher_is_better=False,
        ),
        Quantity(
            "RMSE (Root Mean Squared Error)",
            sklmetrics.mean_squared_error(
                y_true=y_true,
                y_pred=y_pred,
                sample_weight=maybe_sample_weights,
                squared=False,
            ),
            higher_is_better=False,
        ),
        Quantity(
            "MAE (Mean Absolute Error)",
            sklmetrics.mean_absolute_error(
                y_true=y_true, y_pred=y_pred, sample_weight=maybe_sample_weights
            ),
            higher_is_better=False,
        ),
        Quantity(
            "Max Absolute Error",
            sklmetrics.max_error(y_true=y_true, y_pred=y_pred),
            higher_is_better=False,
        ),
        # --- Neutral quantities ---
        Quantity("Number of Samples", len(y_true)),
        Quantity(
            "Sample Weights Used?", "No" if maybe_sample_weights is None else "Yes"
        ),
        Quantity("Mean Ground Truth", np.average(y_true, weights=maybe_sample_weights)),
        Quantity("Mean Prediction", np.average(y_pred, weights=maybe_sample_weights)),
    ]

    # --- Quantities that only make sense without sample weights ---
    if maybe_sample_weights is None:
        quantities.append(
            Quantity(
                "Median Absolute Error",
                sklmetrics.median_absolute_error(y_true=y_true, y_pred=y_pred),
                higher_is_better=False,
            )
        )
        quantities.append(
            Quantity(
                "Median Ground Truth", float(np.median(y_true)), higher_is_better=None
            )
        )
        quantities.append(
            Quantity(
                "Median Prediction", float(np.median(y_pred)), higher_is_better=None
            )
        )

    return quantities
