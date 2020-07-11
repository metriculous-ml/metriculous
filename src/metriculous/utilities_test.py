from typing import Sequence

import numpy as np
import pytest
from sklearn import metrics as sklmetrics

from . import utilities


def test_sample_weights() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
    weights = utilities.sample_weights_simulating_class_distribution(
        y_true=y_true,  # distribution: [0.4, 0.2, 0.4]
        hypothetical_class_distribution=[0.90, 0.08, 0.02],
    )

    expected_weights = np.array(
        [
            0.90 / 0.4,
            0.90 / 0.4,
            0.90 / 0.4,
            0.90 / 0.4,
            0.08 / 0.2,
            0.08 / 0.2,
            0.02 / 0.4,
            0.02 / 0.4,
            0.02 / 0.4,
            0.02 / 0.4,
        ]
    )
    assert np.shape(weights) == np.shape(y_true)
    np.testing.assert_allclose(weights, expected_weights)

    # Now use the sample weights and see if they have the desired effect:
    # Use predictions where first four entries,
    # which correspond to true class 0, are correct.
    some_prediction = np.array([0, 0, 0, 0, 2, 2, 1, 1, 1, 1])
    accuracy_with_weights = sklmetrics.accuracy_score(
        y_true=y_true, y_pred=some_prediction, sample_weight=weights
    )
    accuracy_without_weights = 0.4
    assert accuracy_with_weights == pytest.approx(
        accuracy_without_weights * 0.90 / 0.4, abs=1e-9
    )


def test_sample_weights__distribution_not_normalized() -> None:
    """
    Checks that an exception is raised if the hypothetical class distribution is not
    normalized.
    """
    not_normalized = [0.4, 0.3, 0.1]
    with pytest.raises(AssertionError):
        _ = utilities.sample_weights_simulating_class_distribution(
            y_true=[0, 1, 2, 0, 1, 2], hypothetical_class_distribution=not_normalized
        )


@pytest.mark.parametrize(
    "y_true, hypothetical_class_weights",
    [
        ([0, 0, 1, 3], [0.5, 0.3, 0.1, 0.1]),
        ([0, 0, 1, 3], [0.5, 0.3, 0.2]),
        ([0, 0, 1, 3], [0.5, 0.1, 0.2, 0.1, 0.1]),
        ([0, 1, 2, 3], [0.5, 0.1, 0.2, 0.1, 0.1]),
        ([3], [0.5, 0.3, 0.2]),
        ([0], [0.5, 0.5]),
        ([1], [1.0]),
    ],
)
def test_sample_weights__class_not_represented(
    y_true: Sequence[int], hypothetical_class_weights: Sequence[float]
) -> None:
    """
    Checks that an exception is raised if at least one class is not represented in the
    input.
    """
    np.testing.assert_allclose(sum(hypothetical_class_weights), 1.0)

    with pytest.raises(AssertionError):
        _ = utilities.sample_weights_simulating_class_distribution(
            y_true=y_true, hypothetical_class_distribution=hypothetical_class_weights
        )
