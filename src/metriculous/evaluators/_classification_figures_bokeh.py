from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from assertpy import assert_that
from bokeh import plotting
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    PrintfTickFormatter,
)
from bokeh.plotting import Figure
from sklearn import metrics as sklmetrics
from sklearn.metrics import accuracy_score

from metriculous.evaluators._bokeh_utils import (
    DARK_BLUE,
    FONT_SIZE,
    HISTOGRAM_ALPHA,
    SCATTER_CIRCLES_FILL_ALPHA,
    SCATTER_CIRCLES_LINE_ALPHA,
    TOOLBAR_LOCATION,
    TOOLS,
    add_title_rows,
    apply_default_style,
    scatter_plot_circle_size,
)
from metriculous.evaluators._classification_utils import check_normalization


def _bokeh_output_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title_rows: Sequence[str],
    sample_weights: Optional[np.ndarray] = None,
    x_label_rotation: Union[str, float] = "horizontal",
) -> Figure:
    """
    Creates a scatter plot that contains the same information as a confusion matrix.

    Args:
        y_true:
            1d integer array indicating the reference labels.
        y_pred:
            1d integer array indicating the predictions.
        class_names:
            Sequence of strings corresponding to the classes.
        title_rows:
            Sequence of strings to be used for the chart title.
        sample_weights:
            Sequence of floats to modify the influence of individual samples.
        x_label_rotation:
            Rotation of the class name labels.

    Returns:
        A bokeh figure

    """

    n = len(class_names)

    assert_that(np.shape(y_true)).is_equal_to(np.shape(y_pred))

    if sample_weights is None:
        sample_weights = np.ones_like(y_true)

    assert_that(np.shape(y_true)).is_equal_to(np.shape(sample_weights))

    p = plotting.figure(
        x_range=class_names,
        plot_height=350,
        plot_width=350,
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
    )

    bins = np.arange(0, n + 1, 1)
    normalize = not np.allclose(sample_weights, 1.0)

    # class distribution in prediction
    p.vbar(
        x=class_names,
        top=np.histogram(y_pred, bins=bins, weights=sample_weights, density=normalize)[
            0
        ],
        width=0.85,
        color=DARK_BLUE,
        alpha=HISTOGRAM_ALPHA,
        legend_label="Prediction",
    )

    # class distribution in ground truth
    p.vbar(
        x=class_names,
        top=np.histogram(y_true, bins=bins, weights=sample_weights, density=normalize)[
            0
        ],
        width=0.85,
        alpha=0.6,
        legend_label="Ground Truth",
        fill_color=None,
        line_color="black",
        line_width=2.5,
    )

    add_title_rows(p, title_rows)
    apply_default_style(p)

    p.yaxis.axis_label = "Fraction of Instances" if normalize else "Number of Instances"

    p.xaxis.major_label_orientation = x_label_rotation

    p.xgrid.grid_line_color = None

    # prevent panning to empty regions
    p.x_range.bounds = (-0.5, 0.5 + len(class_names))

    return p


def _bokeh_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title_rows: Sequence[str],
    x_label_rotation: Union[str, float] = "horizontal",
    y_label_rotation: Union[str, float] = "vertical",
) -> Figure:

    """
    Creates a confusion matrix heatmap.

    Args:
        y_true:
            1d integer array indicating the reference labels.
        y_pred:
            1d integer array indicating the predictions.
        class_names:
            Sequence of strings corresponding to the classes.
        title_rows:
            Sequence of strings to be used for the chart title.
        x_label_rotation:
            Rotation of the x-axis class name labels.
        y_label_rotation:
            Rotation of the y-axis class name labels.

    Returns:
        A bokeh figure

    """

    cm = sklmetrics.confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum()
    cm_normalized_by_pred = cm.astype("float") / cm.sum(axis=0, keepdims=True)
    cm_normalized_by_true = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    predicted = list()
    actual = list()
    count = list()
    normalized = list()
    normalized_by_pred = list()
    normalized_by_true = list()

    for i, i_class in enumerate(class_names):
        for j, j_class in enumerate(class_names):
            predicted.append(j_class)
            actual.append(i_class)
            count.append(cm[i, j])
            normalized.append(cm_normalized[i, j])
            normalized_by_pred.append(cm_normalized_by_pred[i, j])
            normalized_by_true.append(cm_normalized_by_true[i, j])

    source = ColumnDataSource(
        data={
            "predicted": predicted,
            "actual": actual,
            "count": count,
            "normalized": normalized,
            "normalized_by_true": normalized_by_true,
            "normalized_by_pred": normalized_by_pred,
        }
    )

    p = plotting.figure(tools=TOOLS, x_range=class_names, y_range=class_names)

    mapper = LinearColorMapper(palette="Viridis256", low=0.0, high=1.0)

    p.rect(
        x="actual",
        y="predicted",
        width=0.95,
        height=0.95,
        source=source,
        fill_color={"field": "normalized_by_true", "transform": mapper},
        line_width=0,
        line_color="black",
    )

    p.xaxis.axis_label = "Ground Truth"
    p.yaxis.axis_label = "Prediction"

    p.xaxis.major_label_orientation = x_label_rotation
    p.yaxis.major_label_orientation = y_label_rotation

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Predicted", "@predicted"),
                ("Ground truth", "@actual"),
                ("Count", "@count"),
                ("Normalized", "@normalized"),
                ("Normalized by prediction", "@normalized_by_pred"),
                ("Normalize by ground truth", "@normalized_by_true"),
            ]
        )
    )

    color_bar = ColorBar(
        color_mapper=mapper,
        major_label_text_font_size=FONT_SIZE,
        ticker=BasicTicker(desired_num_ticks=10),
        formatter=PrintfTickFormatter(format="%.1f"),
        label_standoff=5,
        border_line_color=None,
        location=(0, 0),
    )
    p.add_layout(color_bar, "right")

    add_title_rows(p, title_rows)
    apply_default_style(p)

    return p


def _bokeh_confusion_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title_rows: Sequence[str],
    x_label_rotation: Union[str, float] = "horizontal",
    y_label_rotation: Union[str, float] = "vertical",
) -> Figure:
    """
    Creates a scatter plot that contains the same information as a confusion matrix.

    Args:
        y_true:
            1d integer array indicating the reference labels.
        y_pred:
            1d integer array indicating the predictions.
        class_names:
            Sequence of strings corresponding to the classes.
        title_rows:
            Sequence of strings to be used for the chart title.
        x_label_rotation:
            Rotation of the x-axis class name labels.
        y_label_rotation:
            Rotation of the y-axis class name labels.

    Returns:
        A bokeh figure

    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length!")

    p = plotting.figure(
        x_range=(-0.5, -0.5 + len(class_names)),
        y_range=(-0.5, -0.5 + len(class_names)),
        plot_height=350,
        plot_width=350,
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        match_aspect=True,
    )

    def noise() -> np.ndarray:
        return (np.random.beta(1, 1, size=len(y_true)) - 0.5) * 0.6

    p.scatter(
        x=y_true + noise(),
        y=y_pred + noise(),
        size=scatter_plot_circle_size(
            num_points=len(y_true),
            biggest=4.0,
            smallest=1.0,
            use_smallest_when_num_points_at_least=5000,
        ),
        color=DARK_BLUE,
        fill_alpha=SCATTER_CIRCLES_FILL_ALPHA,
        line_alpha=SCATTER_CIRCLES_LINE_ALPHA,
    )

    add_title_rows(p, title_rows)
    apply_default_style(p)

    p.xaxis.axis_label = "Ground Truth"
    p.yaxis.axis_label = "Prediction"

    arange = np.arange(len(class_names))

    p.xaxis.ticker = arange
    p.yaxis.ticker = arange

    p.xaxis.major_label_overrides = {i: name for i, name in enumerate(class_names)}
    p.yaxis.major_label_overrides = {i: name for i, name in enumerate(class_names)}

    p.xaxis.major_label_orientation = x_label_rotation
    p.yaxis.major_label_orientation = y_label_rotation

    # grid between classes, not at classes
    p.xgrid.ticker = arange[0:-1] + 0.5
    p.ygrid.ticker = arange[0:-1] + 0.5

    p.xgrid.grid_line_width = 3
    p.ygrid.grid_line_width = 3

    # prevent panning to empty regions
    p.x_range.bounds = (-0.5, -0.5 + len(class_names))
    p.y_range.bounds = (-0.5, -0.5 + len(class_names))

    return p


def _bokeh_roc_curve(
    y_true_binary: np.ndarray,
    y_pred_score: np.ndarray,
    title_rows: Sequence[str],
    sample_weights: Optional[np.ndarray],
) -> Figure:
    """Plots an interactive receiver operator characteristic (ROC) curve.

    Args:
        y_true_binary:
            An array of zeros and ones.
        y_pred_score:
            A continuous value, such as a probability estimate for the positive class.
        title_rows:
            Sequence of strings to be used for the chart title.
        sample_weights:
            Sequence of floats to modify the influence of individual samples.

    Returns:
        A bokeh figure

    """
    assert y_true_binary.shape == y_pred_score.shape
    assert set(y_true_binary).issubset({0, 1}) or set(y_true_binary).issubset(
        {False, True}
    )
    assert np.ndim(y_true_binary) == 1

    fpr, tpr, thresholds = sklmetrics.roc_curve(
        y_true=y_true_binary, y_score=y_pred_score, sample_weight=sample_weights
    )

    source = ColumnDataSource(
        data={"FPR": fpr, "TPR": tpr, "threshold": thresholds, "specificity": 1.0 - fpr}
    )

    p = plotting.figure(
        plot_height=400,
        plot_width=350,
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        # toolbar_location=None,  # hides entire toolbar
        match_aspect=True,
    )

    p.xaxis.axis_label = "FPR"
    p.yaxis.axis_label = "TPR"

    add_title_rows(p, title_rows)
    apply_default_style(p)

    curve = p.line(x="FPR", y="TPR", line_width=2, color=DARK_BLUE, source=source)
    p.line(
        x=[0.0, 1.0], y=[0.0, 1.0], line_alpha=0.75, color="grey", line_dash="dotted"
    )

    p.add_tools(
        HoverTool(
            # make sure there is no tool tip for the diagonal baseline
            renderers=[curve],
            tooltips=[
                ("TPR", "@TPR"),
                ("FPR", "@FPR"),
                ("Sensitivity", "@TPR"),
                ("Specificity", "@specificity"),
                ("Threshold", "@threshold"),
            ],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode="vline",
        )
    )

    return p


def _bokeh_precision_recall_curve(
    y_true_binary: np.ndarray,
    y_pred_score: np.ndarray,
    title_rows: Sequence[str],
    sample_weights: Optional[np.ndarray],
) -> Figure:
    """
    Plots an interactive precision recall curve.

    Args:
        y_true_binary:
            An array of zeros and ones.
        y_pred_score:
            A continuous value, such as a probability estimate for the positive class.
        title_rows:
            Sequence of strings to be used for the chart title.
        sample_weights:
            Sequence of floats to modify the influence of individual samples.

    Returns:
        A bokeh figure

    """
    assert y_true_binary.shape == y_pred_score.shape
    assert set(y_true_binary).issubset({0, 1}) or set(y_true_binary).issubset(
        {False, True}
    )
    assert np.ndim(y_true_binary) == 1

    # Note: len(thresholds) == len(precision) - 1
    # The last precision recall pair does not have a corresponding threshold.
    precision, recall, thresholds = sklmetrics.precision_recall_curve(
        y_true=y_true_binary, probas_pred=y_pred_score, sample_weight=sample_weights
    )
    precision = precision[:-1]
    recall = recall[:-1]

    p = plotting.figure(
        plot_height=400,
        plot_width=350,
        x_range=(-0.05, 1.05),
        y_range=(-0.05, 1.05),
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        # match_aspect=True,
    )

    source = ColumnDataSource(
        data={"precision": precision, "recall": recall, "threshold": thresholds}
    )

    # reminder: tpr == recall == sensitivity
    p.line(x="recall", y="precision", line_width=2, source=source)

    add_title_rows(p, title_rows)
    apply_default_style(p)

    p.xaxis.axis_label = "Recall"
    p.yaxis.axis_label = "Precision"

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Precision", "@precision"),
                ("Recall", "@recall"),
                ("Threshold", "@threshold"),
            ],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode="vline",
        )
    )

    return p


def _bokeh_automation_rate_analysis(
    y_target_one_hot: np.ndarray,
    y_pred_proba: np.ndarray,
    title_rows: Sequence[str],
    sample_weights: Optional[np.ndarray],
) -> Figure:
    """
    Plots various quantities over automation rate, where a single probability threshold
    is used for all classes to decide if we are confident enough to automate the
    classification.

    Args:
        y_target_one_hot:
            Array with one-hot encoded ground truth, shape(n_samples, n_classes).
        y_pred_proba:
            Array with estimated probability distributions, shape(n_samples, n_classes).
        title_rows:
            Sequence of strings to be used for the chart title.
        sample_weights:
            Sequence of floats to modify the influence of individual samples.

    Returns:
        A bokeh figure

    """
    # ----- check input -----
    assert y_target_one_hot.ndim == 2
    assert y_pred_proba.ndim == 2
    assert (
        y_target_one_hot.shape == y_pred_proba.shape
    ), f"{y_target_one_hot.shape} != {y_pred_proba.shape}"
    check_normalization(y_target_one_hot, axis=1)
    check_normalization(y_pred_proba, axis=1)
    assert set(y_target_one_hot.ravel()) == {0, 1}, set(y_target_one_hot.ravel())

    if sample_weights is None:
        sample_weights = np.ones(len(y_target_one_hot))

    assert_that(sample_weights.shape).is_equal_to((len(y_target_one_hot),))

    # ----- compute chart data -----
    y_target = y_target_one_hot.argmax(axis=1)
    argmaxes = y_pred_proba.argmax(axis=1)
    maxes = y_pred_proba.max(axis=1)
    assert isinstance(maxes, np.ndarray)  # making IntelliJ's type checker happy

    chart_data: Dict[str, List[float]] = {
        "automation_rate": [],
        "threshold": [],
        "accuracy": [],
    }

    for threshold in sorted(maxes):
        automated = maxes >= threshold
        chart_data["automation_rate"].append(
            np.average(automated, weights=sample_weights)
        )
        chart_data["threshold"].append(threshold)
        chart_data["accuracy"].append(
            accuracy_score(
                y_true=y_target[automated],
                y_pred=argmaxes[automated],
                sample_weight=sample_weights[automated],
            )
        )

    # ----- bokeh plot -----
    p = plotting.figure(
        plot_height=400,
        plot_width=350,
        x_range=(-0.05, 1.05),
        y_range=(-0.05, 1.05),
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        # match_aspect=True,
    )

    source = ColumnDataSource(
        data={key: np.array(lst) for key, lst in chart_data.items()}
    )

    accuracy_line = p.line(
        x="automation_rate",
        y="accuracy",
        line_width=2,
        color=DARK_BLUE,
        source=source,
        legend_label="Accuracy",
    )

    p.line(
        x="automation_rate",
        y="threshold",
        line_width=2,
        color="grey",
        source=source,
        legend_label="Threshold",
    )

    # make sure something is visible if lines consist of just a single point
    p.scatter(
        x=source.data["automation_rate"][[0, -1]], y=source.data["accuracy"][[0, -1]]
    )
    p.scatter(
        x=source.data["automation_rate"][[0, -1]],
        y=source.data["threshold"][[0, -1]],
        color="grey",
    )

    add_title_rows(p, title_rows)
    apply_default_style(p)

    p.xaxis.axis_label = "Automation Rate"
    p.legend.location = "bottom_left"

    p.add_tools(
        HoverTool(
            renderers=[accuracy_line],
            tooltips=[
                ("Accuracy", "@accuracy"),
                ("Threshold", "@threshold"),
                ("Automation Rate", "@automation_rate"),
            ],
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode="vline",
        )
    )

    return p
