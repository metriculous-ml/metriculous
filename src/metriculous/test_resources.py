from typing import List
from typing import Tuple

import numpy as np


def noisy_prediction(targets_one_hot: np.array, noise_factor: float):
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
) -> Tuple[np.ndarray, List[np.ndarray]]:
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
