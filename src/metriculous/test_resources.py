from pathlib import Path
from typing import Sequence, Tuple

import jupytext
import numpy as np


def noisy_prediction(targets_one_hot: np.ndarray, noise_factor: float) -> np.ndarray:
    """Simulates a classifier prediction on the dataset."""
    assert targets_one_hot.ndim == 2
    # Add some noise to the predictions to simulate a classifier
    noisy_target = targets_one_hot + noise_factor * np.random.random(
        size=targets_one_hot.shape
    )
    # Normalize the rows, making sure they are valid probability distributions
    probability_distributions = noisy_target / noisy_target.sum(axis=1, keepdims=True)
    return probability_distributions


def generate_input(
    num_classes: int, num_samples: int, num_models: int
) -> Tuple[np.ndarray, Sequence[np.ndarray]]:
    target_class_indices = np.random.randint(0, high=num_classes, size=num_samples)
    targets_one_hot = np.eye(num_classes)[target_class_indices]

    # For each model that goes into the comparison, let's generate a prediction.
    # Note that we pick a random noise factor to make sure some models have more noise
    # than others.
    predicted_probabilities = [
        noisy_prediction(targets_one_hot, noise_factor=3 * np.random.random())
        for i_model in range(num_models)
    ]
    return targets_one_hot, predicted_probabilities


def check_that_ipynb_and_py_files_are_synchronized(
    ipynb_file: Path, py_file: Path
) -> None:
    py_from_ipynb = jupytext.writes(jupytext.read(ipynb_file), fmt="py:percent")
    py_from_py = jupytext.writes(jupytext.read(py_file), fmt="py:percent")
    # Remove commas, as there seems to be some conflict with black's format and jupytext
    assert py_from_ipynb.replace(",", "") == py_from_py.replace(",", "")
