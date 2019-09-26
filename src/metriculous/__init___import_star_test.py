from metriculous import *  # noqa


def test_import_star():
    _ = Quantity("q", 42.0)  # noqa

    e = Evaluator()  # noqa
    _ = Evaluation([], "MyModel")  # noqa

    _ = Comparator(evaluator=e)  # noqa
    _ = Comparison([])  # noqa

    _ = evaluators.ClassificationEvaluator()  # noqa
