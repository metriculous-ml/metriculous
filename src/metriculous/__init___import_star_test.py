from metriculous import *  # noqa


def test_import_star() -> None:
    _ = Quantity("q", 42.0)  # noqa

    _ = Evaluator()  # noqa
    _ = Evaluation("MyModel", [], [])  # noqa

    classification_evaluator = evaluators.ClassificationEvaluator()  # noqa

    _ = evaluators.RegressionEvaluator()  # noqa

    _ = Comparison([])  # noqa
