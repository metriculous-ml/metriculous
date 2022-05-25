from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from bokeh import plotting
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import Figure
from sklearn.utils import check_consistent_length, column_or_1d

from metriculous.evaluators.bokeh_utils import (
    DARK_BLUE,
    SCATTER_CIRCLES_FILL_ALPHA,
    SCATTER_CIRCLES_LINE_ALPHA,
    TOOLBAR_LOCATION,
    TOOLS,
    add_title_rows,
    apply_default_style,
)


def _bokeh_probability_calibration_plot(
    y_true_binary: np.ndarray,
    y_pred_score: np.ndarray,
    title_rows: Sequence[str],
    # TODO sample_weights: Optional[np.ndarray],
) -> Callable[[], Figure]:
    """Probability calibration plot.

    Args:
        y_true_binary:
            A 1D array of zeros and ones for the ground truth.
        y_pred_score:
            A 1D array of continuous values, such as a probability estimate for the positive class.
        title_rows:
            Sequence of strings to be used for the chart title.

    Returns:
        A callable that returns a fresh bokeh figure each time it is called

    """

    def figure() -> Figure:
        assert y_true_binary.shape == y_pred_score.shape
        assert set(y_true_binary).issubset({0, 1}) or set(y_true_binary).issubset(
            {False, True}
        )
        assert np.ndim(y_true_binary) == 1
        assert min(y_pred_score) >= 0.0, min(y_pred_score)
        assert max(y_pred_score) <= 1.0, max(y_pred_score)

        n_bins = min(100, int(len(y_true_binary) / 10))
        calibration_curve = _calibration_curve(
            y_true_binary=y_true_binary,
            y_pred_score=y_pred_score,
            n_bins=n_bins,
            strategy="quantile",
        )

        source = ColumnDataSource(
            data={
                "positive_fractions": calibration_curve.positive_fractions,
                "pred_confidence_bin_means": calibration_curve.pred_confidence_bin_means,
                "bin_counts": calibration_curve.bin_counts,
                "bin_edges": list(
                    zip(
                        calibration_curve.bin_edges_left,
                        calibration_curve.bin_edges_right,
                    )
                ),
            }
        )

        p = plotting.figure(
            # plot_height=370,
            # plot_width=350,
            x_range=(-0.05, 1.05),
            y_range=(-0.05, 1.05),
            tools=TOOLS,
            toolbar_location=TOOLBAR_LOCATION,
            match_aspect=True,
        )

        # Draw a diagonal to indicate what perfect calibration would look like
        p.line(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            line_alpha=0.75,
            color="grey",
            line_dash="dotted",
        )

        circles = p.circle(
            source=source,
            x="pred_confidence_bin_means",
            y="positive_fractions",
            size=3.0,
            color=DARK_BLUE,
            fill_alpha=SCATTER_CIRCLES_FILL_ALPHA,
            line_alpha=SCATTER_CIRCLES_LINE_ALPHA,
        )

        add_title_rows(p, title_rows)
        apply_default_style(p)

        p.xaxis.axis_label = "Predicted probability"
        p.yaxis.axis_label = "Actual fraction of positive samples"

        p.add_tools(
            HoverTool(
                # Make sure there is no tool tip for the diagonal baseline
                renderers=[circles],
                tooltips=[
                    ("Predicted (bin mean)", "@pred_confidence_bin_means"),
                    ("Positive fraction", "@positive_fractions"),
                    ("Samples in bin", "@bin_counts"),
                ],
                # display a tooltip whenever the cursor is vertically in line with a glyph
                mode="vline",
            )
        )

        return p

    return figure


@dataclass(frozen=True)
class _CalibrationCurve:
    positive_fractions: np.ndarray
    pred_confidence_bin_means: np.ndarray
    bin_counts: np.ndarray
    bin_edges_left: np.ndarray
    bin_edges_right: np.ndarray

    def __post_init__(self) -> None:
        assert (
            self.positive_fractions.shape == self.pred_confidence_bin_means.shape
        ), self
        assert self.positive_fractions.shape == self.bin_counts.shape, self
        assert self.bin_edges_left.shape == self.bin_counts.shape, self
        assert self.bin_edges_right.shape == self.bin_counts.shape, self


def _calibration_curve(
    y_true_binary: np.ndarray, y_pred_score: np.ndarray, n_bins: int, strategy: str
) -> _CalibrationCurve:
    """
    Compute true and predicted probabilities for a calibration curve.
    Copied and modified from https://github.com/scikit-learn/scikit-learn/blob/80598905e/sklearn/calibration.py#L873

    Args:
        y_true_binary:
            A 1D array of zeros and ones for the ground truth.
        y_pred_score:
            A 1D array of probability estimates for the positive class.
        n_bins:
            Number of bins to discretize the [0, 1] interval. A bigger number
            requires more data. Bins with no samples (i.e. without
            corresponding values in `y_prob`) will not be returned, thus the
            returned arrays may have less than `n_bins` values.
        strategy:
            {'uniform', 'quantile'}, default='uniform'
            Strategy used to define the widths of the bins.
            uniform: The bins have identical widths.
            quantile: The bins have the same number of samples and depend on `y_prob`.

    """

    y_true_binary = column_or_1d(y_true_binary)
    y_pred_score = column_or_1d(y_pred_score)
    check_consistent_length(y_true_binary, y_pred_score)

    assert y_true_binary.shape == y_pred_score.shape
    assert set(y_true_binary).issubset({0, 1}) or set(y_true_binary).issubset(
        {False, True}
    )
    assert min(y_pred_score) >= 0.0, min(y_pred_score)
    assert max(y_pred_score) <= 1.0, max(y_pred_score)

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_pred_score, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.digitize(y_pred_score, bins) - 1

    bin_pred_sums = np.bincount(binids, weights=y_pred_score, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true_binary, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_pred_sums[nonzero] / bin_total[nonzero]

    bin_edges_left = bins[nonzero]
    bin_edges_right = np.array(
        [bins[i + 1] for i, _ in enumerate(nonzero[:-1]) if nonzero[i]]
    )

    return _CalibrationCurve(
        positive_fractions=prob_true,
        pred_confidence_bin_means=prob_pred,
        bin_counts=bin_total[nonzero],
        bin_edges_left=bin_edges_left,
        bin_edges_right=bin_edges_right,
    )
