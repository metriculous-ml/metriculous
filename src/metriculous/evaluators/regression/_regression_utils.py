from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np

Floats = Union[Sequence[float], np.ndarray]


@dataclass
class RegressionData:
    targets: np.ndarray
    predictions: np.ndarray

    def __init__(
        self, targets: Floats, predictions: Floats,
    ):
        self.targets = np.asarray(targets)
        self.predictions = np.asarray(predictions)

    def __post_init__(self) -> None:
        assert isinstance(self.targets, np.ndarray)
        assert isinstance(self.predictions, np.ndarray)

        if self.targets.ndim != 1:
            raise ValueError(
                f"For regression problems metriculous expects one-dimensional sequences of floats,"
                f"but you have provided a ground truth object of shape {self.targets.shape}"
            )

        if self.predictions.ndim != 1:
            raise ValueError(
                f"For regression problems metriculous expects one-dimensional sequences of floats,"
                f"but you have provided a prediction object of shape {self.predictions.shape}"
            )

        if self.targets.shape != self.predictions.shape:
            raise ValueError(
                f"For regression problems metriculous expects "
                f"ground truth and prediction sequences of the same length, "
                f"but the ground truth has length {len(self.targets)}"
                f"while the prediction has length {len(self.predictions)}"
            )
