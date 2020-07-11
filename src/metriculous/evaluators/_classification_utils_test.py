import numpy as np
import pytest
from numpy import testing as npt

from metriculous.evaluators._classification_utils import (
    ClassificationData,
    ProbabilityMatrix,
)


class TestProbabilityMatrix:
    def test_that_it_can_be_initialized_from_nested_list(self) -> None:
        _ = ProbabilityMatrix(
            distributions=[[0.8, 0.1, 0.1], [0.0, 0.9, 0.1], [0.2, 0.2, 0.6]]
        )

    def test_that_it_can_be_initialized_from_array(self) -> None:
        _ = ProbabilityMatrix(distributions=np.eye(5))

    def test_that_it_raises_error_if_not_normalized(self) -> None:

        with pytest.raises(AssertionError, match="1.0"):
            _ = ProbabilityMatrix(distributions=[[0.9, 0.11, 0.0]])

        with pytest.raises(AssertionError, match="1.0"):
            _ = ProbabilityMatrix(distributions=[[0.9, 0.0, 0.0]])

        with pytest.raises(AssertionError, match="1.0"):
            _ = ProbabilityMatrix(distributions=[[]])

    def test_that_it_raises_error_if_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="two-dimensional"):
            _ = ProbabilityMatrix(distributions=[])

        with pytest.raises(ValueError, match="two-dimensional") as info:
            # noinspection PyTypeChecker
            _ = ProbabilityMatrix(distributions=[[[]]])  # type: ignore
        assert info.match("Expected a two-dimensional")
        assert info.match("but received array of shape")
        assert info.match("(1, 1, 0)")

    def test_properties(self) -> None:
        pm = ProbabilityMatrix(
            distributions=[
                [0.8, 0.1, 0.1],
                [0.2, 0.2, 0.6],
                [0.0, 0.9, 0.1],
                [0.0, 0.9, 0.1],
            ]
        )
        assert pm.n_classes == 3
        assert pm.n_samples == 4
        npt.assert_equal(pm.argmaxes, desired=[0, 2, 1, 1])
        npt.assert_equal(
            pm.argmaxes_one_hot,
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
        )


class TestClassificationData:
    def test_that_it_does_not_smoke(self) -> None:
        cd = ClassificationData(
            target=ProbabilityMatrix(
                distributions=[
                    [0.8, 0.1, 0.1],
                    [0.2, 0.2, 0.6],
                    [0.0, 0.9, 0.1],
                    [0.0, 0.9, 0.1],
                ]
            ),
            pred=ProbabilityMatrix(
                distributions=[
                    [0.8, 0.1, 0.1],
                    [0.2, 0.2, 0.6],
                    [0.0, 0.9, 0.1],
                    [0.0, 0.9, 0.1],
                ]
            ),
        )

        assert cd.n_samples == 4
        assert cd.n_classes == 3

    def test_that_it_raises_if_n_samples_different(self) -> None:
        with pytest.raises(ValueError, match="samples") as info:
            _ = ClassificationData(
                target=ProbabilityMatrix(distributions=[[0.8, 0.1, 0.1]]),
                pred=ProbabilityMatrix(
                    distributions=[[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]]
                ),
            )

        assert info.match("1")
        assert info.match("2")
        assert info.match("samples")

    def test_that_it_raises_if_n_classes_different(self) -> None:
        with pytest.raises(ValueError, match="classes") as info:
            _ = ClassificationData(
                target=ProbabilityMatrix(
                    distributions=[[0.8, 0.1, 0.1, 0.0], [0.8, 0.1, 0.1, 0.0]]
                ),
                pred=ProbabilityMatrix(
                    distributions=[[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]]
                ),
            )

        assert info.match("4 classes")
        assert info.match("3 classes")
