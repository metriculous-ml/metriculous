from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from bokeh.layouts import column, gridplot
from bokeh.models import Column, LayoutDOM
from bokeh.plotting import Figure, figure

from metriculous.evaluators._bokeh_utils import (
    DARK_BLUE,
    DARK_RED,
    HISTOGRAM_ALPHA,
    SCATTER_CIRCLES_FILL_ALPHA,
    SCATTER_CIRCLES_LINE_ALPHA,
    TOOLS,
    apply_default_style,
    scatter_plot_circle_size,
    title_div,
)
from metriculous.evaluators._regression_utils import Floats, RegressionData

GROUND_TRUTH_HISTOGRAM_ENVELOPE_COLOR = "black"
GROUND_TRUTH_HISTOGRAM_ENVELOPE_ALPHA = 0.7

DASHED_LINE_PROPERTIES = dict(color="black", alpha=0.6, line_dash=[6, 2], width=1)

DEFAULT_N_HISTOGRAM_BINS = 20


def _bokeh_scatter_with_histograms(
    x: Floats,
    y: Floats,
    x_label: str,
    y_label: str,
    x_histogram_fill_color: Optional[str],
    y_histogram_fill_color: Optional[str],
    y_histogram_envelope_color: Optional[str],
    title_rows: Sequence[str],
    n_bins: int,
) -> Tuple[LayoutDOM, Figure]:
    """
    Scatter plot with small histograms attached to the axes.
    """
    assert len(x) == len(y)

    # --- Create the scatter plot ---
    p_scatter = figure(
        tools=TOOLS,
        plot_width=250,
        plot_height=250,
        min_border=0,
        min_border_left=20,
        sizing_mode="stretch_width",
    )

    p_scatter.scatter(
        x,
        y,
        color=DARK_BLUE,
        size=(
            scatter_plot_circle_size(
                len(x),
                biggest=3,
                smallest=1,
                use_smallest_when_num_points_at_least=5000,
            )
        ),
        fill_alpha=SCATTER_CIRCLES_FILL_ALPHA,
        line_alpha=SCATTER_CIRCLES_LINE_ALPHA,
    )

    p_scatter.xaxis.axis_label = x_label
    p_scatter.yaxis.axis_label = y_label

    # --- Compute histograms ---
    hist_x, hist_x_edges = np.histogram(x, bins=n_bins)
    hist_y, hist_y_edges = np.histogram(y, bins=n_bins)
    # Make sure the two histograms are similarly scaled
    hist_range = (-0.1, 1.05 * max(max(hist_x), max(hist_y)))

    # --- Create the horizontal (ground truth) histogram, above the scatter plot ---
    p_hist_x_above = figure(
        plot_width=p_scatter.plot_width,
        plot_height=30,
        x_range=p_scatter.x_range,
        y_range=hist_range,
        min_border=0,
        min_border_left=0,
        x_axis_location=None,
        y_axis_location=None,
        sizing_mode="stretch_width",
        tools=TOOLS,
    )
    p_hist_x_above.quad(
        bottom=0,
        left=hist_x_edges[:-1],
        right=hist_x_edges[1:],
        top=hist_x,
        color=x_histogram_fill_color,
        alpha=HISTOGRAM_ALPHA,
        line_color="white",
    )

    # --- Create the vertical (prediction) histogram, to the right of the scatter plot ---
    p_hist_y_right = figure(
        plot_width=40,
        plot_height=p_scatter.plot_height,
        x_range=hist_range,
        y_range=p_scatter.y_range,
        x_axis_location=None,
        y_axis_location=None,
        sizing_mode="fixed",
        tools=TOOLS,
    )
    p_hist_y_right.quad(
        left=0,
        bottom=hist_y_edges[:-1],
        top=hist_y_edges[1:],
        right=hist_y,
        color=y_histogram_fill_color,
        alpha=HISTOGRAM_ALPHA,
        line_color="white" if y_histogram_fill_color is not None else None,
    )

    if y_histogram_envelope_color is not None:
        p_hist_y_right.line(
            x=[0.0] + [h for h in hist_y for _ in range(2)] + [0.0],
            y=[e for e in hist_y_edges for _ in range(2)],
            color=GROUND_TRUTH_HISTOGRAM_ENVELOPE_COLOR,
            alpha=GROUND_TRUTH_HISTOGRAM_ENVELOPE_ALPHA,
        )

    # --- Style and layout ---
    apply_default_style(p_scatter)
    apply_default_style(p_hist_x_above)
    apply_default_style(p_hist_y_right)
    # Overwrite some of the defaults
    p_hist_x_above.ygrid.grid_line_color = None
    p_hist_y_right.xgrid.grid_line_color = None

    grid = gridplot(
        # fmt:off
        [
            [p_hist_x_above, None],
            [p_scatter, p_hist_y_right]
        ],
        merge_tools=True,
        toolbar_location="right",
        toolbar_options={
            "logo": None,
        },
        sizing_mode="scale_width",
        # fmt:on
    )

    layout = column(title_div(title_rows), grid, sizing_mode="scale_width")

    return layout, p_scatter


def _bokeh_actual_vs_predicted_scatter_with_histograms(  # TODO rename
    data: RegressionData,
    title_rows: Sequence[str],
    n_bins: int = DEFAULT_N_HISTOGRAM_BINS,
) -> Callable[[], LayoutDOM]:
    """
    Plots ground truth and prediction in a scatter plot with small histograms attached to the axes.
    """

    def function() -> LayoutDOM:
        # --- Create the scatter plot ---
        layout, p_scatter = _bokeh_scatter_with_histograms(
            x=data.predictions,
            y=data.targets,
            x_label="Predicted",
            y_label="Actual",
            x_histogram_fill_color=DARK_BLUE,
            y_histogram_fill_color=None,
            y_histogram_envelope_color=GROUND_TRUTH_HISTOGRAM_ENVELOPE_COLOR,
            title_rows=title_rows,
            n_bins=n_bins,
        )

        # Dashed line to indicate the identity
        min_target = min(data.targets)
        max_target = max(data.targets)
        p_scatter.line(
            x=[min_target, max_target],
            y=[min_target, max_target],
            **DASHED_LINE_PROPERTIES,
        )

        return layout

    return function


def _bokeh_residual_vs_predicted_scatter_with_histograms(
    data: RegressionData,
    title_rows: Sequence[str],
    n_bins: int = DEFAULT_N_HISTOGRAM_BINS,
) -> Callable[[], LayoutDOM]:
    """
    Plots ground truth and prediction in a scatter plot with small histograms attached to the axes.
    """
    predictions = data.predictions
    residuals = np.asarray(data.targets) - np.asarray(predictions)

    def function() -> LayoutDOM:
        # --- Create the scatter plot ---
        layout, p_scatter = _bokeh_scatter_with_histograms(
            x=predictions,
            y=residuals,
            x_label="Predicted",
            y_label="Residual",
            title_rows=title_rows,
            n_bins=n_bins,
            x_histogram_fill_color=DARK_BLUE,
            y_histogram_fill_color=DARK_RED,
            y_histogram_envelope_color=None,
        )

        # Dashed line to indicate zero
        p_scatter.line(
            x=[min(predictions), max(predictions)], y=[0, 0], **DASHED_LINE_PROPERTIES
        )

        return layout

    return function


def _bokeh_targets_and_predictions_histograms(
    data: RegressionData,
    title_rows: Sequence[str],
    n_bins: int = DEFAULT_N_HISTOGRAM_BINS,
) -> Callable[[], Column]:
    """
    Plots two histograms, one for the ground truth and one for the prediction.
    """
    targets = data.targets
    predictions = data.predictions

    def function() -> Column:

        p = figure(tools=TOOLS)

        # Plot looks better when we use the same bin sizes for both histograms.
        _, edges = np.histogram(list(targets) + list(predictions), bins=n_bins)

        pred_hist, pred_edges = np.histogram(predictions, bins=edges)
        np.testing.assert_allclose(pred_edges, edges)
        p.quad(
            top=pred_hist,
            bottom=0,
            left=pred_edges[:-1],
            right=pred_edges[1:],
            fill_color=DARK_BLUE,
            fill_alpha=0.5,
            line_color="white",
            line_alpha=0.5,
            legend_label="Predicted  ",
        )

        target_hist, target_edges = np.histogram(targets, bins=edges)
        np.testing.assert_allclose(target_edges, edges)
        p.line(
            x=[e for e in target_edges for _ in range(2)],
            y=[0.0] + [h for h in target_hist for _ in range(2)] + [0.0],
            color=GROUND_TRUTH_HISTOGRAM_ENVELOPE_COLOR,
            alpha=GROUND_TRUTH_HISTOGRAM_ENVELOPE_ALPHA,
            legend_label="Actual  ",
        )

        # Increase the y range a bit to leave enough space for the legend
        p.y_range.end = 1.25 * max(max(target_hist), max(pred_hist))

        apply_default_style(p)
        p.xaxis.axis_label = "Value"
        p.yaxis.axis_label = "Number of Occurrences"
        p.legend.padding = 4
        p.legend.orientation = "horizontal"

        return column(title_div(title_rows), p, sizing_mode="scale_width")

    return function


def _bokeh_residual_histogram(
    data: RegressionData,
    title_rows: Sequence[str],
    n_bins: int = DEFAULT_N_HISTOGRAM_BINS,
) -> Callable[[], Column]:
    """
    Plots a histogram of the errors.
    """
    targets = data.targets
    predictions = data.predictions

    def function() -> Column:

        p = figure(tools=TOOLS)

        residuals = np.asarray(targets) - np.asarray(predictions)

        residual_hist, edges = np.histogram(residuals, bins=n_bins)

        p.quad(
            top=residual_hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color=DARK_RED,
            fill_alpha=0.5,
            line_color="white",
            line_alpha=0.5,
        )

        apply_default_style(p)
        p.xaxis.axis_label = "Residual"
        p.yaxis.axis_label = "Number of Occurrences"

        return column(title_div(title_rows), p, sizing_mode="scale_width")

    return function
