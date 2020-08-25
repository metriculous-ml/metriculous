import numpy as np
from bokeh.embed import file_html
from bokeh.resources import CDN

from metriculous.evaluators._classification_figures_bokeh import _bokeh_confusion_matrix


def test_bokeh_confusion_matrix__does_not_crash_when_class_is_never_predicted() -> None:
    lazy_figure = _bokeh_confusion_matrix(
        y_true=np.asarray([0, 1, 2]),
        y_pred=np.asarray([1, 1, 1]),
        class_names=["A", "B", "C"],
        title_rows=["Some", "Title"],
    )
    figure = lazy_figure()

    _ = file_html(figure, resources=CDN)
