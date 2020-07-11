import numpy as np
import pytest

from .. import Quantity
from . import RegressionEvaluator


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
        for quantity in evaluation.quantities:
            assert isinstance(quantity, Quantity)
            assert isinstance(quantity.value, (float, str, int))
        for lazy_figure in evaluation.lazy_figures:
            _ = lazy_figure()
