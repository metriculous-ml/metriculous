import numpy as np
import numpy.testing as npt
import pytest
from bokeh.embed import file_html
from bokeh.resources import CDN
from sklearn.metrics import accuracy_score as sklearn_accuracy

from metriculous.evaluators._classification_figures_bokeh import (
    _bokeh_confusion_matrix,
    _faster_accuracy,
)


def test_bokeh_confusion_matrix__does_not_crash_when_class_is_never_predicted() -> None:
    lazy_figure = _bokeh_confusion_matrix(
        y_true=np.asarray([0, 1, 2]),
        y_pred=np.asarray([1, 1, 1]),
        class_names=["A", "B", "C"],
        title_rows=["Some", "Title"],
    )
    figure = lazy_figure()

    _ = file_html(figure, resources=CDN)


@pytest.mark.parametrize("use_sample_weights", [False, True])
def test_faster_accuracy(use_sample_weights: bool) -> None:
    n_samples = 500
    for i in range(10):
        rng = np.random.RandomState(seed=i)
        y_true = rng.randint(0, 10, size=n_samples)
        y_pred = rng.randint(0, 10, size=n_samples)
        sample_weights = rng.random(size=n_samples) if use_sample_weights else None
        npt.assert_allclose(
            actual=_faster_accuracy(y_true, y_pred, sample_weights=sample_weights),
            desired=sklearn_accuracy(y_true, y_pred, sample_weight=sample_weights),
        )
