from typing import Optional

import pytest

from metriculous import Quantity


@pytest.mark.parametrize("name", ["", "accuracy", "What Ever"])
@pytest.mark.parametrize("value", [-0.5, 0.0, 1e15])
@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("description", [None, "", "Quantifies the whateverness"])
def test_quantity(
    name: str, value: float, higher_is_better: bool, description: Optional[str]
) -> None:
    quantity = Quantity(name, value, higher_is_better, description)
    quantity_ = Quantity(name, value, higher_is_better, description)
    assert quantity == quantity_
