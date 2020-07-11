from typing import Optional

import numpy as np
from bokeh import plotting
from bokeh.layouts import column
from bokeh.models import Title
from bokeh.plotting import Figure

TOOLS = "pan,box_zoom,reset"
TOOLBAR_LOCATION = "right"


def _bokeh_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_label: int,
    class_name: Optional[str] = None,
) -> Figure:
    """
    Creates heatmaps of the predictions and ground_truth
    corresponding to the class_label

    Args:
        y_true:
            3d integer array indicating the ground_truth masks.
            Shape: (Num_Samples, Height, Width)
        y_pred:
            3d integer array indicating the predictions of the model as the same shape
            as y_true
        class_label:
            An integer corresponding to the class for which the heatmap is desired
        class_name:
            Class Name corresponding to the class_label

    Returns:
        A bokeh figure

    """

    if y_pred.shape != y_true.shape:
        raise ValueError(
            (
                "The shapes of y_pred and y_true must be the same. "
                f"Got y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}"
            )
        )

    if class_label not in np.unique(y_true):
        raise ValueError("Incorrect class_label provided, doesn't exist in y_true")

    if class_name is None:
        class_name = f"Class {class_label}"

    padding = 5

    mean_activation_predictions = np.average(
        (y_pred == class_label).astype(np.uint8), axis=0
    )

    mean_activation_ground_truth = np.average(
        (y_true == class_label).astype(np.uint8), axis=0
    )

    p1 = plotting.figure(
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        width=y_true.shape[2],
        height=y_true.shape[1],
    )
    p1.x_range.range_padding = p1.y_range.range_padding = 0
    p1.toolbar.logo = None

    p1.image(
        image=[mean_activation_predictions],
        x=0,
        y=0,
        dw=y_true.shape[2],
        dh=y_true.shape[1],
    )

    p1.add_layout(Title(text="Ground Truth", align="center"), "below")
    p1.add_layout(
        Title(text=f"Heatmap for {class_name}", align="center"), place="above"
    )
    p1.axis.visible = False

    p2 = plotting.figure(
        tools=TOOLS,
        toolbar_location=TOOLBAR_LOCATION,
        width=y_true.shape[2],
        height=y_true.shape[1],
        x_range=p1.x_range,
    )
    p2.x_range.range_padding = p2.y_range.range_padding = 0
    p2.toolbar.logo = None

    p2.image(
        image=[mean_activation_ground_truth],
        x=0,
        y=y_true.shape[1] + padding,
        dw=y_true.shape[2],
        dh=y_true.shape[1],
    )

    p2.add_layout(Title(text="Prediction", align="center"), "below")

    p2.axis.visible = False

    return column(p1, p2)
