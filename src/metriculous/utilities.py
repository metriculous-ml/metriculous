from typing import Sequence, Union

import numpy as np
import numpy.testing as npt
from assertpy import assert_that


def sample_weights_simulating_class_distribution(
    y_true: Union[Sequence[int], np.ndarray],
    hypothetical_class_distribution: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    """
    Computes a 1D array of sample weights that results in the requested
    `hypothetical_class_distribution` if applied to the dataset. This is useful when you
    know that the class distribution in your dataset deviates from the distribution you
    expect to encounter in the environment where your machine learning model is going to
    be deployed.

    Example:
        You have a data set with 40% spam 60% ham emails. However, you expect that
        only 4% of the emails in the deployment environment will be spam, and you would
        like to measure various performance characteristics on a dataset with 4% spam
        and 96% ham. This function will return an array with
            * sample weights  4% / 40% = 0.1 for all of the spam examples
            * sample weights 96% / 60% = 1.6 for all of the ham examples
        if called with:
            >>> weights = sample_weights_simulating_class_distribution(
            ...    y_true=[0, 1, 1, 0, 1, 0, 1, 1, 0, 1],  # zeros for spam
            ...    hypothetical_class_distribution=[0.04, 0.96]
            ... )
            >>> print(weights)
            array([0.1 , 1.6])

    Args:
        y_true:
            1D array of integers with class indices of the dataset. There must be at
            least one sample for each class.
        hypothetical_class_distribution:
            Sequence of floats describing the distribution you assume to encounter in
            your deployment environment.

    Returns:
        1D numpy array with sample weights, same length as `y_true`.

    """
    # --- check input ---
    assert_that(set(y_true)).is_equal_to(
        set(range(len(hypothetical_class_distribution)))
    )

    assert_that(len(set(y_true))).is_equal_to(len(hypothetical_class_distribution))

    y_true = np.asarray(y_true)
    hypothetical_class_distribution = np.asarray(hypothetical_class_distribution)

    npt.assert_allclose(
        hypothetical_class_distribution.sum(),
        1.0,
        err_msg="Probability distribution does not sum up to 1.0",
    )
    assert_that(y_true.ndim).is_equal_to(1)
    assert_that(hypothetical_class_distribution.ndim).is_equal_to(1)

    # --- compute output ---
    class_distribution = np.bincount(y_true) / len(y_true)
    npt.assert_equal(class_distribution > 0.0, True)
    npt.assert_allclose(class_distribution.sum(), 1.0)

    weights = [
        hypothetical_class_distribution[y] / class_distribution[y] for y in y_true
    ]
    return np.array(weights)
