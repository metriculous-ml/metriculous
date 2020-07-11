from dataclasses import replace
from typing import Callable, List, Optional, Sequence

import numpy as np
import pytest

from .._evaluation import Evaluation, Quantity
from ..evaluators import ClassificationEvaluator
from ..test_resources import noisy_prediction


def random_targets_one_hot(num_classes: int, num_samples: int) -> np.ndarray:
    target_class_indices = np.random.randint(0, high=num_classes, size=num_samples)
    return np.eye(num_classes)[target_class_indices]


@pytest.mark.parametrize("noise_factor, num_samples", [(0.1, 100), (10.0, 200)])
@pytest.mark.parametrize(
    "classes, simulated_class_distribution",
    [
        (None, None),
        (["Cat", "Dog", "Lion"], [0.01, 0.02, 0.97]),
        (["Cat", "Dog", "Lion"], None),
        (["Spam", "Ham"], [0.2, 0.8]),
    ],
)
@pytest.mark.parametrize(
    argnames=(
        "one_vs_all_quantities,"
        "one_vs_all_figures,"
        "top_n_accuracies,"
        "filter_quantities,"
        "primary_metric,"
    ),
    argvalues=zip(
        [False, True],
        [True, False],
        [(4,), [], [2, 3, 42]],
        [None, lambda name: "a" in name, lambda name: False],
        ["Accuracy", None, None],
    ),
)
@pytest.mark.parametrize("use_sample_weights", [False, True])
def test_ClassificationEvaluator(
    noise_factor: float,
    simulated_class_distribution: Optional[Sequence[float]],
    num_samples: int,
    classes: Optional[List[str]],
    one_vs_all_quantities: bool,
    one_vs_all_figures: bool,
    top_n_accuracies: Sequence[int],
    filter_quantities: Callable[[str], bool],
    primary_metric: Optional[str],
    use_sample_weights: bool,
) -> None:
    """Basic smoke test making sure we don't crash with valid input."""

    np.random.seed(42)

    targets_one_hot = random_targets_one_hot(
        num_classes=len(classes) if classes is not None else 3, num_samples=num_samples
    )
    prediction = noisy_prediction(targets_one_hot, noise_factor=noise_factor)

    ce = ClassificationEvaluator(
        class_names=classes,
        one_vs_all_quantities=one_vs_all_quantities,
        one_vs_all_figures=one_vs_all_figures,
        top_n_accuracies=top_n_accuracies,
        filter_quantities=filter_quantities,
        primary_metric=primary_metric,
        simulated_class_distribution=(
            None if use_sample_weights else simulated_class_distribution
        ),
    )

    evaluation = ce.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
        sample_weights=(
            42.0 * np.random.random(size=num_samples)
            if use_sample_weights is True
            else None
        ),
    )

    evaluation.figures()

    assert isinstance(evaluation, Evaluation)
    assert evaluation.model_name == "MockModel"


@pytest.mark.parametrize("num_samples", [100, 200, 999])
@pytest.mark.parametrize(
    "use_sample_weights, simulated_class_distribution",
    [(False, None), (False, [0.3, 0.5, 0.2]), (True, None)],
)
def test_ClassificationEvaluator_perfect_prediction(
    num_samples: int,
    use_sample_weights: bool,
    simulated_class_distribution: List[float],
) -> None:
    np.random.seed(42)
    targets_one_hot = random_targets_one_hot(num_classes=3, num_samples=num_samples)
    prediction = noisy_prediction(targets_one_hot, noise_factor=0.0)
    ce = ClassificationEvaluator(
        simulated_class_distribution=simulated_class_distribution
    )
    evaluation = ce.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
        sample_weights=(
            42.0 * np.random.random(size=num_samples)
            if use_sample_weights is True
            else None
        ),
    )
    assert isinstance(evaluation, Evaluation)
    assert evaluation.model_name == "MockModel"

    expected_quantities = [
        Quantity(name="Accuracy", value=1.0, higher_is_better=True, description=None),
        Quantity(
            name="ROC AUC Macro Average",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="ROC AUC Micro Average",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="F1-Score Macro Average",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="F1-Score Micro Average",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="ROC AUC class_0 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="ROC AUC class_1 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="ROC AUC class_2 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="Average Precision class_0 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="Average Precision class_1 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="Average Precision class_2 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="F1-Score class_0 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="F1-Score class_1 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="F1-Score class_2 vs Rest",
            value=1.0,
            higher_is_better=True,
            description=None,
        ),
        Quantity(
            name="Mean KLD(P=target||Q=prediction)",
            value=0.0,
            higher_is_better=False,
            description=None,
        ),
        Quantity(
            name="Log Loss",
            value=2.1094237467877998e-15,
            higher_is_better=False,
            description=None,
        ),
        Quantity(
            name="Brier Score Loss", value=0.0, higher_is_better=False, description=None
        ),
        Quantity(
            name="Brier Score Loss (Soft Targets)",
            value=0.0,
            higher_is_better=False,
            description=None,
        ),
        Quantity(
            name="Max Entropy", value=0.0, higher_is_better=None, description=None
        ),
        Quantity(
            name="Mean Entropy", value=0.0, higher_is_better=None, description=None
        ),
        Quantity(
            name="Min Entropy", value=0.0, higher_is_better=None, description=None
        ),
        Quantity(
            name="Max Probability", value=1.0, higher_is_better=None, description=None
        ),
        Quantity(
            name="Min Probability", value=0.0, higher_is_better=None, description=None
        ),
    ]

    assert len(evaluation.quantities) == len(expected_quantities)
    for actual, expected in zip(evaluation.quantities, expected_quantities):
        # check that everything except value is equal
        assert replace(actual, value=42) == replace(expected, value=42)
        # check that values are approximately equal
        if isinstance(expected.value, str):
            assert isinstance(actual, str)
            assert actual.value == expected.value
        else:
            assert isinstance(expected.value, float)
            assert isinstance(actual.value, float)
            np.testing.assert_allclose(actual.value, expected.value)


@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize(
    "quantity_filter",
    [
        lambda name: False,
        lambda name: True,
        lambda name: "F1" not in name,
        lambda name: "vs Rest" not in name,
    ],
)
def test_ClassificationEvaluator_filter_quantities(
    num_samples: int, quantity_filter: Callable[[str], bool]
) -> None:
    np.random.seed(42)
    targets_one_hot = random_targets_one_hot(num_classes=3, num_samples=num_samples)
    prediction = noisy_prediction(targets_one_hot, noise_factor=0.0)

    ce_all = ClassificationEvaluator()
    ce_filtering = ClassificationEvaluator(filter_quantities=quantity_filter)

    evaluation_all = ce_all.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
    )

    evaluation_filtered = ce_filtering.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
    )

    # assert all equal except quantities
    # (ignore figures as they do not support equality in the way we need it)
    assert replace(evaluation_all, quantities=[], lazy_figures=[]) == replace(
        evaluation_filtered, quantities=[], lazy_figures=[]
    )

    for quantity in evaluation_all.quantities:
        if quantity_filter(quantity.name):
            same_quantity = evaluation_filtered.get_by_name(quantity.name)
            assert same_quantity == quantity
        else:
            with pytest.raises(ValueError):
                evaluation_filtered.get_by_name(quantity.name)

    for filtered_quantity in evaluation_filtered.quantities:
        same_quantity = evaluation_all.get_by_name(filtered_quantity.name)
        assert same_quantity == filtered_quantity


@pytest.mark.parametrize("num_samples", [100, 200])
@pytest.mark.parametrize(
    "desired_number_of_figures, figure_filter",
    [
        (0, lambda name: False),
        (10, None),
        (10, lambda name: True),
        (9, lambda name: "Distribution" not in name),
        (4, lambda name: "vs Rest" not in name),
    ],
)
def test_ClassificationEvaluator_filter_figures(
    num_samples: int,
    desired_number_of_figures: int,
    figure_filter: Callable[[str], bool],
) -> None:
    np.random.seed(42)
    targets_one_hot = random_targets_one_hot(num_classes=3, num_samples=num_samples)
    prediction = noisy_prediction(targets_one_hot, noise_factor=0.0)

    ce_all = ClassificationEvaluator(one_vs_all_figures=True)
    ce_filtering = ClassificationEvaluator(
        one_vs_all_figures=True, filter_figures=figure_filter
    )

    evaluation_all = ce_all.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
    )

    evaluation_filtered = ce_filtering.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
    )

    # assert all equal except figures
    assert replace(evaluation_all, lazy_figures=[]) == replace(
        evaluation_filtered, lazy_figures=[]
    )

    # check number of figures
    assert len(evaluation_filtered.lazy_figures) == desired_number_of_figures


@pytest.mark.parametrize("num_samples", [100, 200])
def test_ClassificationEvaluator_exception_when_passing_distribution_and_weights(
    num_samples: int,
) -> None:
    """
    Checks that an exception is raised when `sample_weights` are passed to an evaluator
    that has been initialized with `simulated_class_distribution`.
    """
    np.random.seed(42)
    targets_one_hot = random_targets_one_hot(num_classes=3, num_samples=num_samples)
    prediction = noisy_prediction(targets_one_hot, noise_factor=0.0)

    ce = ClassificationEvaluator(
        one_vs_all_figures=True, simulated_class_distribution=[0.3, 0.1, 0.6]
    )

    _ = ce.evaluate(
        ground_truth=targets_one_hot,
        model_prediction=prediction,
        model_name="MockModel",
    )

    with pytest.raises(AssertionError) as exception_info:
        _ = ce.evaluate(
            ground_truth=targets_one_hot,
            model_prediction=prediction,
            model_name="MockModel",
            sample_weights=np.random.random(size=len(targets_one_hot)),
        )

    assert str(exception_info.value) == (
        "Cannot use `sample_weights` with ClassificationEvaluator that"
        " was initialized with `simulated_class_distribution`."
    )
