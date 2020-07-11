import copy
from dataclasses import replace
from typing import Callable, List, Tuple

import numpy as np
import pytest

from .._evaluation import Evaluation, Quantity
from ..evaluators import SegmentationEvaluator


def get_random_prediction_and_mask(
    image_size: Tuple[int, int, int], num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.random.randint(0, num_classes, image_size),
        np.random.randint(0, num_classes, image_size),
    )


@pytest.mark.parametrize("classes", (["dog", "cat", "snake"], ["dog", "cat"]))
def test_SegmentationEvaluator(classes: List[str]) -> None:

    np.random.seed(42)

    num_classes = len(classes)

    prediction, mask = get_random_prediction_and_mask((2, 256, 256), num_classes)

    se = SegmentationEvaluator(num_classes, class_names=classes)

    evaluation = se.evaluate(
        ground_truth=mask, model_prediction=prediction, model_name="MockModel"
    )

    assert isinstance(evaluation, Evaluation)
    assert evaluation.model_name == "MockModel"


@pytest.mark.parametrize("classes", (["dog", "cat", "snake"], ["dog", "cat"]))
def test_SegmentationEvaluator_perfect_prediction(classes: List[str]) -> None:

    np.random.seed(42)

    num_classes = len(classes)

    predictions, _ = get_random_prediction_and_mask((2, 256, 256), num_classes)
    mask = copy.deepcopy(predictions)

    se = SegmentationEvaluator(num_classes, class_names=classes)

    evaluation = se.evaluate(
        ground_truth=mask, model_prediction=predictions, model_name="MockModel"
    )

    evaluation.figures()

    expected_quantities = []

    for class_name in classes:
        expected_quantities.append(
            Quantity(name=f"{class_name} mIoU", value=1.0, higher_is_better=True)
        )

    expected_quantities.append(
        Quantity(name="Class weighted Mean mIoU", value=1.0, higher_is_better=True)
    )

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


@pytest.mark.parametrize(
    "num_classes, class_names", [(1, ["dog", "cat"]), (2, ["dog"])]
)
def test_SegmentationEvaluator_inconsistent_class_names(
    num_classes: int, class_names: List[str]
) -> None:

    """
    Tests if the __init__ method of SegmentationEvaluator raises an error if the
    length of the class_names list is not equal to num_classes

    """

    with pytest.raises(ValueError):
        _ = SegmentationEvaluator(num_classes, class_names=class_names)


@pytest.mark.parametrize("num_classes, class_weights", [(1, [0.2, 0.3]), (2, [0.2])])
def test_SegmentationEvaluator_inconsistent_class_weights(
    num_classes: int, class_weights: List[float]
) -> None:

    """
    Tests if the __init__ method of SegmentationEvaluator raises an error if the
    length of the class_weights list is not equal to num_classes

    """

    with pytest.raises(ValueError):
        _ = SegmentationEvaluator(num_classes, class_weights=class_weights)


@pytest.mark.parametrize(
    "num_classes, ground_truth, model_prediction",
    [
        (3, *get_random_prediction_and_mask((2, 256, 256), 2)),
        (2, *get_random_prediction_and_mask((2, 256, 256), 3)),
    ],
)
def test_SegmentationEvaluator_inconsistent_num_classes(
    num_classes: int, ground_truth: np.ndarray, model_prediction: np.ndarray
) -> None:
    """
    Tests if the evaluate method of SegmentationEvaluator raises an error if the
    actual number of classes present in the ground_truth/prediction is not equal to
    num_classes.

    """

    se = SegmentationEvaluator(num_classes)

    with pytest.raises(ValueError):
        se.evaluate(ground_truth, model_prediction, model_name="MockModel")


@pytest.mark.parametrize(
    "num_classes, ground_truth, model_prediction",
    [
        (
            3,
            np.random.randint(0, 3, (1, 256, 256)),
            np.random.randint(0, 3, (2, 256, 256)),
        )
    ],
)
def test_SegmentationEvaluator_inconsistent_shapes(
    num_classes: int, ground_truth: np.ndarray, model_prediction: np.ndarray
) -> None:
    """
    Tests if the evaluate method of SegmentationEvaluator raises an error if the
    shapes of the ground_truth and model_prediction aren't the same

    """

    se = SegmentationEvaluator(num_classes)

    with pytest.raises(ValueError):
        se.evaluate(ground_truth, model_prediction, model_name="MockModel")


@pytest.mark.parametrize(
    "num_classes, ground_truth, model_prediction",
    [
        (
            3,
            np.random.randint(0, 3, (256, 256)),
            np.random.randint(0, 3, (2, 256, 256)),
        ),
        (
            3,
            np.random.randint(0, 3, (2, 256, 256)),
            np.random.randint(0, 3, (256, 256)),
        ),
    ],
)
def test_SegmentationEvaluator_not_a_3D_array(
    num_classes: int, ground_truth: np.ndarray, model_prediction: np.ndarray
) -> None:
    """
    Tests if the evaluate method of SegmentationEvaluator raises an error if the
    ground_truth or model_prediction isn't a 3D array

    """

    se = SegmentationEvaluator(num_classes)

    with pytest.raises(ValueError):
        se.evaluate(ground_truth, model_prediction, model_name="MockModel")


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize(
    "quantity_filter",
    [
        lambda name: False,
        lambda name: True,
        lambda name: "Weighted" not in name,
        lambda name: "mIoU" not in name,
    ],
)
def test_SegmentationEvaluator_filter_quantities(
    num_classes: int, quantity_filter: Callable[[str], bool]
) -> None:
    np.random.seed(42)
    predictions, mask = get_random_prediction_and_mask((2, 256, 256), num_classes)

    se_all = SegmentationEvaluator(num_classes)
    se_filtering = SegmentationEvaluator(num_classes, filter_quantities=quantity_filter)

    evaluation_all = se_all.evaluate(
        ground_truth=mask, model_prediction=predictions, model_name="MockModel"
    )

    evaluation_filtered = se_filtering.evaluate(
        ground_truth=mask, model_prediction=predictions, model_name="MockModel"
    )

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


@pytest.mark.parametrize(
    "num_classes, desired_number_of_figures, figure_filter",
    [
        (3, 0, lambda name: False),
        (3, 4, lambda name: True),
        (3, 1, lambda name: "Heatmap" not in name),
        (3, 3, lambda name: "Class" not in name),
        (2, 2, lambda name: "Class" not in name),
        (2, 3, lambda name: True),
    ],
)
def test_SegmentationEvaluator_filter_figures(
    num_classes: int,
    desired_number_of_figures: int,
    figure_filter: Callable[[str], bool],
) -> None:

    np.random.seed(42)
    predictions, mask = get_random_prediction_and_mask((2, 256, 256), num_classes)

    se_all = SegmentationEvaluator(num_classes)
    se_filtering = SegmentationEvaluator(num_classes, filter_figures=figure_filter)

    evaluation_all = se_all.evaluate(
        ground_truth=mask, model_prediction=predictions, model_name="MockModel"
    )

    evaluation_filtered = se_filtering.evaluate(
        ground_truth=mask, model_prediction=predictions, model_name="MockModel"
    )

    assert replace(evaluation_all, lazy_figures=[]) == replace(
        evaluation_filtered, lazy_figures=[]
    )

    assert len(evaluation_filtered.lazy_figures) == desired_number_of_figures
