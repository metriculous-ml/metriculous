"""Module defining generic metric functions."""
from typing import List, Optional, Tuple

import numpy as np
from assertpy import assert_that
from sklearn.metrics import roc_auc_score, roc_curve


def normalized(matrix: np.ndarray) -> np.ndarray:
    """Returns normalized array where each row sums up to 1.0."""
    assert np.ndim(matrix) == 2
    sums = np.sum(matrix, axis=1, keepdims=True)
    # avoid crash on zeros
    matrix = matrix + (sums == 0.0) * 1e-15
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def cross_entropy(
    target_probas: np.ndarray, pred_probas: np.ndarray, epsilon: float = 1e-15
) -> float:
    """Returns the cross-entropy for probabilistic ground truth labels.

    Args:
        target_probas: 2D array with rows being target probability distributions.
        pred_probas: 2D array with rows being estimated probability distributions.
        epsilon: Clipping offset to avoid numerical blowup (NaNs, inf, etc).
    """
    # check normalization before clipping
    assert np.allclose(
        np.sum(target_probas, axis=1), 1.0, atol=1e-3
    ), "Target probability distributions not normalized!"
    assert np.allclose(
        np.sum(pred_probas, axis=1), 1.0, atol=1e-3
    ), "Predicted probability distributions not normalized!"

    # clip predicted probabilities
    pred_probas = np.clip(pred_probas, a_min=epsilon, a_max=1.0 - epsilon)
    # normalize
    pred_probas = normalized(pred_probas)
    # compute cross entropy
    values = -np.sum(target_probas * np.log(pred_probas), axis=1)
    # noinspection PyTypeChecker
    ce: float = np.mean(values)
    return ce


def a_vs_b_auroc(
    target_ints: np.ndarray, predicted_probas: np.ndarray, class_a: int, class_b: int
) -> Optional[float]:
    """
    Keeps only targets of class A or B, then computes the ROC AUC for the
    binary problem.

    Args:
        target_ints: 1d array of target class integers.
        predicted_probas: 2d array of predicted probabilities, one row per data point.
        class_a: Integer specifying the positive class.
        class_b: Integer specifying the negative class.

    Returns:
        A float or None if the result could not be computed.

    """
    # only consider instances with targets of class A or B
    filter_mask = np.logical_or(target_ints == class_a, target_ints == class_b)
    target_ints = target_ints[filter_mask]
    predicted_probas = predicted_probas[filter_mask]

    # return None if not both classes represented
    if len(np.unique(target_ints)) != 2:
        return None

    # consider only probability columns for class A and B and renormalize
    binary_probas = normalized(predicted_probas[:, (class_a, class_b)])

    # use class A as the positive class
    scores = binary_probas[:, 0]

    return roc_auc_score(y_true=target_ints == class_a, y_score=scores)


def one_vs_all_auroc_values(
    target_ints: np.ndarray, predicted_probas: np.ndarray
) -> List[Optional[float]]:
    """Returns one AUROC (area under ROC curve, aka ROC AUC) score per class.

    Args:
        target_ints: 1d array of target class integers.
        predicted_probas: 2d array of predicted probabilities, one row per data point.

    Returns:
        A list with one AUROC value per class.

    """
    assert len(predicted_probas) == len(target_ints)

    n_classes = predicted_probas.shape[1]

    auroc_values = []
    for positive_class in range(n_classes):
        scores = predicted_probas[:, positive_class]
        is_positive_class = target_ints == positive_class
        if any(is_positive_class) and not all(is_positive_class):
            auroc_values.append(roc_auc_score(y_true=is_positive_class, y_score=scores))
        else:
            auroc_values.append(None)

    return auroc_values


def sensitivity_at_x_specificity(
    target_ints: np.ndarray, positive_probas: np.ndarray, at_specificity: float
) -> Tuple[Optional[float], Optional[float]]:
    """Compute sensitivity (recall) at a given specificity.

    Sensitivity = true positive rate
                = true positives / positives
                = recall
                = P(prediction positive | class positive)

    Specificity = true negative rate
                = true negatives / negatives
                = 1 - false positive rate
                = P(prediction negative | class negative)

    Args:
        target_ints: 1d array of binary class labels, zeros and ones
        positive_probas: 1d array of probabilities of class 1
        at_specificity: specificity at which to compute sensitivity

    Returns:
        (float): sensitivity at returned specificity
        (float): specificity closest to input specificity

    """
    assert 0 < at_specificity < 1

    if len(set(target_ints)) < 2:
        return None, None

    fprs, sensitivities, _ = roc_curve(target_ints, positive_probas)
    specificities = 1.0 - fprs

    # last and first entries are not interesting (0 or 1)
    if len(specificities) > 2:
        specificities = specificities[1:-1]
        sensitivities = sensitivities[1:-1]

    # find point on curve that is closest to desired at_specificity
    index = np.argmin(np.abs(specificities - at_specificity))
    return sensitivities[index], specificities[index]


def specificity_at_x_sensitivity(
    target_ints: np.ndarray, positive_probas: np.ndarray, at_sensitivity: float
) -> Tuple[Optional[float], Optional[float]]:
    """Compute specificity at a given sensitivity (recall).

    Sensitivity = true positive rate
                = true positives / positives
                = recall
                = P(prediction positive | class positive)

    Specificity = true negative rate
                = true negatives / negatives
                = 1 - false positive rate
                = P(prediction negative | class negative)

    Args:
        target_ints: 1d array of binary class labels, zeros and ones
        positive_probas: 1d array of probabilities of class 1
        at_sensitivity: sensitivity at which to compute specificity

    Returns:
        (float): specificity at returned sensitivity
        (float): sensitivity closest to input sensitivity

    """
    assert 0 < at_sensitivity < 1

    if len(set(target_ints)) < 2:
        return None, None

    fprs, sensitivities, _ = roc_curve(target_ints, positive_probas)
    specificities = 1.0 - fprs

    # last and first entries are not interesting
    if len(specificities) > 2:
        specificities = specificities[1:-1]
        sensitivities = sensitivities[1:-1]

    # find point on curve that is closest to desired sensitivity
    index = np.argmin(np.abs(sensitivities - at_sensitivity))
    return specificities[index], sensitivities[index]


def top_n_accuracy(
    target_ints: np.ndarray,
    predicted_probas: np.ndarray,
    n: int,
    sample_weights: Optional[np.ndarray] = None,
) -> float:
    """Fraction of test cases where the true target is among the top n predictions."""
    assert len(target_ints) == len(predicted_probas)
    assert np.ndim(target_ints) == 1
    assert np.ndim(predicted_probas) == 2
    if sample_weights is None:
        sample_weights = np.ones_like(target_ints)
    assert_that(sample_weights.shape).is_equal_to(target_ints.shape)
    np.testing.assert_array_equal(sample_weights >= 0.0, True)

    # sort predicted class indices by probability (ascending)
    classes_by_probability = predicted_probas.argsort(axis=1)
    # take last n columns, because we sorted ascending
    top_n_predictions = classes_by_probability[:, -n:]

    # check if target is included
    is_target_in_top_n_predictions = [
        target in top_n for target, top_n in zip(target_ints, top_n_predictions)
    ]

    top_n_acc = np.average(is_target_in_top_n_predictions, weights=sample_weights)
    return top_n_acc
