from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
from numpy import testing as npt

NORMALIZATION_ABS_TOLERANCE = 1e-5
NORMALIZATION_REL_TOLERANCE = 1e-5

Integers = Union[Sequence[int], np.ndarray]


def check_normalization(
    probabilities: Union[np.ndarray, Sequence[float]],
    axis: int,
    msg: str = "Probability distribution(s) not normalized",
) -> None:
    npt.assert_allclose(
        np.sum(probabilities, axis=axis),
        desired=1.0,
        rtol=NORMALIZATION_REL_TOLERANCE,
        atol=NORMALIZATION_ABS_TOLERANCE,
        err_msg=msg,
    )


@dataclass
class ProbabilityMatrix:
    proba_matrix: np.ndarray
    argmaxes: np.ndarray
    argmaxes_one_hot: np.ndarray
    n_classes: int
    n_samples: int

    def __init__(self, distributions: Union[Sequence[Sequence[float]], np.ndarray]):
        self.proba_matrix = np.asarray(distributions)
        self._validate_input(self.proba_matrix)
        self.n_samples = self.proba_matrix.shape[0]
        self.n_classes = self.proba_matrix.shape[1]
        self.argmaxes = np.argmax(self.proba_matrix, axis=1)  # type: ignore
        self.argmaxes_one_hot = np.eye(self.n_classes)[self.argmaxes]

    @staticmethod
    def _validate_input(probability_array: np.ndarray) -> None:
        assert isinstance(probability_array, np.ndarray)
        if probability_array.ndim != 2:
            raise ValueError(
                f"Expected a two-dimensional probability distribution array with one row per"
                f" sample, but received array of shape {probability_array.shape}"
            )
        npt.assert_allclose(
            probability_array.sum(axis=1),
            desired=1.0,
            rtol=NORMALIZATION_REL_TOLERANCE,
            atol=NORMALIZATION_ABS_TOLERANCE,
            err_msg=(
                "Rows of probability distribution array are expected to sum up to 1.0,"
                " but they don't"
            ),
        )
        n_negative = np.sum(probability_array < 0.0)
        if n_negative > 0:
            raise ValueError(
                f"Received {n_negative} negative values in probability distribution array."
            )


@dataclass(frozen=True)
class ClassificationData:
    target: ProbabilityMatrix
    pred: ProbabilityMatrix

    def __post_init__(self) -> None:
        if self.target.n_samples != self.pred.n_samples:
            raise ValueError(
                f"Received ground truth data with {self.target.n_samples} samples"
                f" but prediction data with {self.pred.n_samples} samples."
            )

        if self.target.n_classes != self.pred.n_classes:
            raise ValueError(
                f"Received ground truth data with {self.target.n_classes} classes"
                f" but prediction data with {self.pred.n_classes} classes."
            )

    @property
    def n_classes(self) -> int:
        return self.target.n_classes

    @property
    def n_samples(self) -> int:
        return self.target.n_samples
