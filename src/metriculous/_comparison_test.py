from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest
from bokeh import plotting
from bokeh.plotting import Figure

from metriculous import Comparison, Evaluation, Quantity


def make_a_bokeh_figure() -> Figure:
    p = plotting.figure()
    p.line(x=[0, 1, 5, 6], y=[40, 60, 30, 50])
    return p


def make_a_comparison(
    with_quantities: bool, with_figures: bool, n_models: int = 5
) -> Comparison:
    evaluations = [
        Evaluation(
            model_name=f"Model_{i}",
            quantities=[]
            if not with_quantities
            else [
                Quantity("Accuracy", value=1.0 / (i + 1), higher_is_better=True),
                Quantity("Error", value=1.0 / (i + 1), higher_is_better=False),
                Quantity("Mean", value=1.0 / (i + 1), higher_is_better=None),
            ],
            lazy_figures=[]
            if not with_figures
            else [make_a_bokeh_figure, make_a_bokeh_figure],
        )
        for i in range(n_models)
    ]
    return Comparison(evaluations)


class TestComparison:
    @pytest.mark.parametrize("with_quantities", [True, False])
    @pytest.mark.parametrize("with_figures", [True, False])
    @pytest.mark.parametrize("include_spacer", [True, False, None])
    def test_html_smoke_test(
        self, with_quantities: bool, with_figures: bool, include_spacer: Optional[bool]
    ) -> None:
        comparison = make_a_comparison(with_quantities, with_figures=with_figures)
        html_string = comparison.html(include_spacer)
        assert isinstance(html_string, str)
        # Check that it can be called a second time. A second call can crash
        # when Bokeh figure objects aren't recreated in each call.
        html_string_2 = comparison.html(include_spacer)
        # HTML will not be exactly equal as bokeh and/or pandas generate
        # different IDs each time
        assert html_string[:100] == html_string_2[:100]
        assert html_string[-100:] == html_string_2[-100:]

    @pytest.mark.parametrize("with_quantities", [True, False])
    @pytest.mark.parametrize("with_figures", [True, False])
    @pytest.mark.parametrize("include_spacer", [True, False, None])
    def test_save_html(
        self, with_quantities: bool, with_figures: bool, include_spacer: Optional[bool]
    ) -> None:
        comparison = make_a_comparison(with_quantities, with_figures=with_figures)
        path = Path("save_html_smoke_test_output.html")
        comparison.save_html(path, include_spacer=include_spacer)
        with path.open() as html_file:
            html_in_file = html_file.read()
        path.unlink()
        for evaluation in comparison.evaluations:
            for q in evaluation.quantities:
                assert q.name in html_in_file

    def test_display_then_html_then_save_html_smoke_test(self) -> None:
        """ Checks that subsequent calls do not interfere with each other. """
        comparison = make_a_comparison(with_quantities=True, with_figures=True)
        comparison.display()
        comparison.html()
        path = Path("test_output.html")
        comparison.save_html(path)
        path.unlink()

    def test_generate_default_html_file_name(self) -> None:
        comparison = make_a_comparison(with_quantities=True, with_figures=True)
        n_models = 5
        name = comparison._generate_default_html_file_name()
        assert name.startswith("comparison_")
        now = datetime.now()
        assert now.strftime("%Y-%m-%d-%H%M") in name
        assert (
            f"{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}{now.minute:02}"
            in name
        )
        assert name.endswith(f"_{n_models}_models.html")
