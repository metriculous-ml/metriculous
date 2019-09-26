import pytest
from assertpy import assert_that


def test_exposed_entities():
    expected = [
        "Comparator",
        "Comparison",
        "Evaluator",
        "Evaluation",
        "Quantity",
        "evaluators",
        "utilities",
    ]

    import metriculous

    assert_that(metriculous.__all__).is_equal_to(expected)


def test_imports_from_style():
    from metriculous import Comparator
    from metriculous import Comparison
    from metriculous import Evaluation
    from metriculous import Evaluator
    from metriculous import Quantity

    num_classes = 42

    _ = Quantity("q", 42.0)

    e = Evaluator()
    _ = Evaluation("MyModel", [], [])

    _ = Comparator(evaluator=e)
    _ = Comparison([])

    with pytest.raises(ImportError):
        # noinspection PyUnresolvedReferences,PyProtectedMember
        from metriculous import ClassificationEvaluator

        _ = ClassificationEvaluator()

    from metriculous.evaluators import ClassificationEvaluator

    _ = ClassificationEvaluator()

    with pytest.raises(ImportError):
        # noinspection PyUnresolvedReferences,PyProtectedMember
        from metriculous import SegmentationEvaluator

        _ = SegmentationEvaluator(num_classes)

    from metriculous.evaluators import SegmentationEvaluator

    _ = SegmentationEvaluator(num_classes)


def test_imports_prefix_style():
    import metriculous as met

    num_classes = 42

    _ = met.Quantity("q", 42.0)

    e = met.Evaluator()
    _ = met.Evaluation("MyModel", [], [])

    _ = met.Comparator(evaluator=e)
    _ = met.Comparison([])

    _ = met.evaluators.ClassificationEvaluator()
    _ = met.evaluators.SegmentationEvaluator(num_classes)

    _ = met.utilities.sample_weights_simulating_class_distribution(
        [0, 1, 2, 2], [0.8, 0.2, 0.0]
    )

    with pytest.raises(AttributeError):
        # noinspection PyUnresolvedReferences
        _ = met.ClassificationEvaluator()
        _ = met.SegmentationEvaluator(num_classes)
