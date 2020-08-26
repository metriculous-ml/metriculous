from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import numpy.testing as npt
import pytest

from .. import Evaluation, Quantity
from . import RegressionEvaluator
from ._bokeh_utils import check_that_all_figures_can_be_rendered


class TestRegressionEvaluator:
    @pytest.mark.parametrize("use_sample_weights", [True, False])
    def test_that_it_does_not_smoke(self, use_sample_weights: bool) -> None:
        n = 100
        evaluation = RegressionEvaluator().evaluate(
            ground_truth=np.random.randn(n),
            model_prediction=np.random.randn(n),
            model_name="Mock Regression Model",
            sample_weights=np.random.random(size=n) if use_sample_weights else None,
        )
        check_any_evaluation(evaluation)

    @pytest.mark.parametrize("use_sample_weights", [True, False])
    def test_perfect_prediction(self, use_sample_weights: bool) -> None:
        n_samples = 100
        ground_truth = np.random.randn(n_samples)
        maybe_sample_weights = (
            np.random.random(size=n_samples) if use_sample_weights else None
        )

        evaluation = RegressionEvaluator().evaluate(
            ground_truth=ground_truth,
            model_prediction=ground_truth,
            model_name="Mock Regression Model",
            sample_weights=maybe_sample_weights,
        )

        check_any_evaluation(evaluation)

        for q in evaluation.quantities:
            print(q)

        for q in evaluation.quantities:
            if "error" in q.name.lower():
                assert q.higher_is_better is False
                assert q.value == 0.0

        expected_quantities = [
            Quantity(name="R^2", value=1.0, higher_is_better=True, description=None),
            Quantity(name="Explained Variance", value=1.0, higher_is_better=True),
            Quantity(
                name="MSE (Mean Squared Error)", value=0.0, higher_is_better=False
            ),
            Quantity(
                name="RMSE (Root Mean Squared Error)", value=0.0, higher_is_better=False
            ),
            Quantity(
                name="MAE (Mean Absolute Error)", value=0.0, higher_is_better=False
            ),
            Quantity(name="Max Absolute Error", value=0.0, higher_is_better=False),
            Quantity(name="Number of Samples", value=n_samples, higher_is_better=None),
            Quantity(
                name="Sample Weights Used?",
                value="Yes" if use_sample_weights else "No",
                higher_is_better=None,
            ),
            Quantity(
                name="Mean Ground Truth",
                value=np.average(ground_truth, weights=maybe_sample_weights),
                higher_is_better=None,
            ),
            Quantity(
                name="Mean Prediction",
                value=np.average(ground_truth, weights=maybe_sample_weights),
                higher_is_better=None,
            ),
        ]

        if not use_sample_weights:
            expected_quantities.extend(
                [
                    Quantity(
                        name="Median Absolute Error", value=0.0, higher_is_better=False
                    ),
                    Quantity(
                        name="Median Ground Truth",
                        value=np.median(ground_truth),
                        higher_is_better=None,
                    ),
                    Quantity(
                        name="Median Prediction",
                        value=np.median(ground_truth),
                        higher_is_better=None,
                    ),
                ]
            )

        assert evaluation.quantities == expected_quantities

    @pytest.mark.parametrize("use_sample_weights", [True, False])
    def test_imperfect_prediction(self, use_sample_weights: bool) -> None:
        n_samples = 100
        ground_truth = np.random.randn(n_samples)
        residual = np.random.randn(n_samples)
        prediction = ground_truth - residual
        maybe_sample_weights = (
            np.random.random(size=n_samples) if use_sample_weights else None
        )

        evaluation = RegressionEvaluator().evaluate(
            ground_truth=ground_truth,
            model_prediction=prediction,
            model_name="Mock Regression Model",
            sample_weights=maybe_sample_weights,
        )

        check_any_evaluation(evaluation)

        for q in evaluation.quantities:
            print(q)

        for q in evaluation.quantities:
            if "error" in q.name.lower():
                assert q.higher_is_better is False
                assert isinstance(q.value, float)
                assert q.value > 0.0

        sample_weights = (
            np.ones_like(ground_truth)
            if maybe_sample_weights is None
            else maybe_sample_weights
        )

        r2_numerator = np.sum(residual ** 2 * sample_weights)

        r2_denominator = np.sum(
            (ground_truth - np.average(ground_truth, weights=maybe_sample_weights)) ** 2
            * sample_weights
        )

        expected_quantities = [
            Quantity(
                name="R^2",
                value=(1.0 - (r2_numerator / r2_denominator)),
                higher_is_better=True,
            ),
            Quantity(
                name="Explained Variance",
                value=(
                    1.0
                    - (
                        var(residual, weights=maybe_sample_weights)
                        / var(ground_truth, weights=maybe_sample_weights)
                    )
                ),
                higher_is_better=True,
            ),
            Quantity(
                name="MSE (Mean Squared Error)",
                value=np.average(residual ** 2, weights=maybe_sample_weights),
                higher_is_better=False,
            ),
            Quantity(
                name="RMSE (Root Mean Squared Error)",
                value=float(
                    np.sqrt(np.average(residual ** 2, weights=maybe_sample_weights))
                ),
                higher_is_better=False,
            ),
            Quantity(
                name="MAE (Mean Absolute Error)",
                value=np.average(np.absolute(residual), weights=maybe_sample_weights),
                higher_is_better=False,
            ),
            Quantity(
                name="Max Absolute Error",
                value=max(np.absolute(residual)),
                higher_is_better=False,
            ),
            Quantity(name="Number of Samples", value=n_samples, higher_is_better=None),
            Quantity(
                name="Sample Weights Used?",
                value="Yes" if use_sample_weights else "No",
                higher_is_better=None,
            ),
            Quantity(
                name="Mean Ground Truth",
                value=np.average(ground_truth, weights=maybe_sample_weights),
                higher_is_better=None,
            ),
            Quantity(
                name="Mean Prediction",
                value=np.average(prediction, weights=maybe_sample_weights),
                higher_is_better=None,
            ),
        ]

        if not use_sample_weights:
            expected_quantities.extend(
                [
                    Quantity(
                        name="Median Absolute Error",
                        value=np.median(np.absolute(residual)),
                        higher_is_better=False,
                    ),
                    Quantity(
                        name="Median Ground Truth",
                        value=np.median(ground_truth),
                        higher_is_better=None,
                    ),
                    Quantity(
                        name="Median Prediction",
                        value=np.median(prediction),
                        higher_is_better=None,
                    ),
                ]
            )

        assert_all_close(
            evaluation.quantities, expected_quantities, atol=1e-10, rtol=1e-10
        )


def check_any_evaluation(evaluation: Evaluation) -> None:
    """ Performs basic checks that should pass for any `Evaluation` object. """
    for quantity in evaluation.quantities:
        assert isinstance(quantity, Quantity)
        assert isinstance(quantity.value, (float, str, int))

    check_that_all_figures_can_be_rendered(evaluation.figures())


def assert_all_close(
    a: Sequence[Quantity], b: Sequence[Quantity], atol: float, rtol: float
) -> None:
    assert len(a) == len(b)
    for qa, qb in zip(a, b):
        if isinstance(qa.value, float):
            npt.assert_allclose(qa.value, qb.value, atol=atol, rtol=rtol)
            assert replace(qa, value="any") == replace(qb, value="any")
        else:
            assert qa == qb


def var(values: np.ndarray, weights: Optional[np.ndarray]) -> float:
    mean = np.average(values, weights=weights)
    return np.average((values - mean) ** 2, weights=weights)
