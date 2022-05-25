import numpy as np
import numpy.testing as npt
import pytest

from metriculous.evaluators.classification.calibration_plot import _calibration_curve


@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_calibration_curve(strategy: str) -> None:
    curve = _calibration_curve(
        y_true_binary=np.array([0, 0, 1, 0, 1, 1]),
        y_pred_score=np.array([0.0, 0.24, 0.48, 0.52, 0.76, 1.0]),
        n_bins=2,
        strategy=strategy,
    )

    assert list(curve.bin_counts) == [3, 3]
    npt.assert_allclose(curve.positive_fractions, [1 / 3, 2 / 3])
    npt.assert_allclose(curve.pred_confidence_bin_means, [0.24, 0.76])
    npt.assert_allclose(curve.bin_edges_left, [0.0, 0.5])
    npt.assert_allclose(curve.bin_edges_right, [0.5, 1.0])


def test_calibration_curve__uniform() -> None:
    curve = _calibration_curve(
        y_true_binary=np.array([0, 1, 1]),
        y_pred_score=np.array([0.3, 0.5, 0.6]),
        n_bins=2,
        strategy="uniform",
    )
    print(f"curve: {curve}")

    assert list(curve.bin_counts) == [2, 1]
    npt.assert_allclose(curve.positive_fractions, [1 / 2, 1 / 1])
    npt.assert_allclose(curve.pred_confidence_bin_means, [0.4, 0.6])
    npt.assert_allclose(curve.bin_edges_left, [0.0, 0.5])
    npt.assert_allclose(curve.bin_edges_right, [0.5, 1.0])


def test_calibration_curve__quantile() -> None:
    curve = _calibration_curve(
        y_true_binary=np.array([0, 1, 1]),
        y_pred_score=np.array([0.3, 0.5, 0.6]),
        n_bins=2,
        strategy="quantile",
    )
    print(f"curve: {curve}")

    assert list(curve.bin_counts) == [1, 2]
    npt.assert_allclose(curve.positive_fractions, [0 / 1, 2 / 2])
    npt.assert_allclose(curve.pred_confidence_bin_means, [0.3, 0.55])
    npt.assert_allclose(curve.bin_edges_left, [0.3, 0.5])
    npt.assert_allclose(curve.bin_edges_right, [0.5, 0.6])
