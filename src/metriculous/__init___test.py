import pytest


def test_exposed_entities() -> None:
    expected = [
        "compare",
        "compare_classifiers",
        "compare_regressors",
        "Comparison",
        "Evaluator",
        "Evaluation",
        "Quantity",
        "utilities",
        "evaluators",
        "ClassificationEvaluator",
        "RegressionEvaluator",
    ]

    import metriculous

    assert metriculous.__all__ == expected


def test_imports_from_style() -> None:
    from metriculous import (
        ClassificationEvaluator,
        Comparison,
        Evaluation,
        Evaluator,
        Quantity,
        RegressionEvaluator,
    )

    _ = Quantity("q", 42.0)

    _ = Evaluator()
    _ = Evaluation("MyModel", [], [])

    _ = Comparison([])

    _ = ClassificationEvaluator()
    _ = RegressionEvaluator()


def test_imports_from_metriculous_evaluators() -> None:

    from metriculous.evaluators import (
        ClassificationEvaluator,
        RegressionEvaluator,
        SegmentationEvaluator,
    )

    _ = ClassificationEvaluator()
    _ = RegressionEvaluator()
    _ = SegmentationEvaluator(num_classes=5)


def test_imports_prefix_style() -> None:
    import metriculous as met

    assert hasattr(met, "compare")

    _ = met.compare_classifiers(
        ground_truth=[[0.8, 0.1, 0.1], [0.0, 0.9, 0.1], [0.2, 0.2, 0.6]],
        model_predictions=[
            [[0.2, 0.2, 0.6], [0.3, 0.4, 0.3], [0.3, 0.4, 0.3]],
            [[0.3, 0.0, 0.7], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]],
        ],
    )

    _ = met.compare_regressors(
        ground_truth=[0.5, 42.0, -3],
        model_predictions=[[0.5, 42.0, -3], [0.5, 42.0, -3]],
    )

    _ = met.Quantity("q", 42.0)

    _ = met.Evaluator()
    _ = met.Evaluation("MyModel", [], [])

    _ = met.Comparison([])

    _ = met.ClassificationEvaluator()
    _ = met.evaluators.ClassificationEvaluator()

    _ = met.RegressionEvaluator()
    _ = met.evaluators.RegressionEvaluator()

    _ = met.evaluators.SegmentationEvaluator(num_classes=42)

    _ = met.utilities.sample_weights_simulating_class_distribution(
        [0, 1, 2, 2], [0.8, 0.2, 0.0]
    )

    with pytest.raises(AttributeError):
        # noinspection PyUnresolvedReferences
        _ = met.SegmentationEvaluator(num_classes=42)  # type: ignore
