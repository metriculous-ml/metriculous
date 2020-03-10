from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional, Union
from typing import List
from typing import Sequence

from IPython.display import HTML
from IPython.display import display
from assertpy import assert_that
from bokeh.embed import file_html
import bokeh.layouts
from bokeh.layouts import column
from bokeh.models import Spacer
from bokeh.resources import CDN
import numpy as np
import pandas as pd

from metriculous._evaluation import Evaluation
from metriculous._evaluation import Evaluator


@dataclass(frozen=True)
class Comparison:
    evaluations: List[Evaluation]

    def __post_init__(self):
        _check_consistency(self.evaluations)

    def display(self, include_spacer=False):
        # noinspection PyTypeChecker
        display(HTML(_html_comparison_table(self.evaluations, include_spacer)))

        # noinspection PyBroadException
        try:
            os.system('say "Model comparison is ready."')
        except Exception:
            pass

    def html(self, include_spacer=False) -> str:
        css = """
        <style>
            html {
              font-family: Verdana;
            }  
            table {
              border-collapse: collapse;
              width: 100%;
            }
            
            th, td {
              font-size: 0.8em;
              padding: 8px;
              text-align: right;
              border-bottom: 1px solid #ddd;
            }
            
            tr:hover {background-color:#f5f5f5;}
        </style>
        """
        return css + _html_comparison_table(self.evaluations, include_spacer)

    def save_html(self, file_path: Union[str, Path], include_spacer=False):
        file_path = Path(file_path)
        if file_path.exists():
            raise FileExistsError(f"Path exists, refusing to overwrite '{file_path}'")
        html_string = self.html(include_spacer)
        with file_path.open(mode="w"):
            file_path.write_text(html_string)


class Comparator:
    """Can generate model comparisons after initialization with an Evaluator."""

    def __init__(self, evaluator: Evaluator):
        self.evaluator = evaluator

    def compare(
        self,
        ground_truth: Any,
        model_predictions: Sequence[Any],
        model_names=None,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Comparison:
        """Generates a Comparison from a list of predictions and the ground truth.

        Args:
            model_predictions:
                List with one prediction object per model to be compared.
            ground_truth:
                A single ground truth object.
            model_names:
                Optional list of model names. If `None` generic names will be generated.
            sample_weights:
                Optional sequence of floats to modify the influence of individual
                samples on the statistics that will be measured.

        Returns:
            A Comparison object with one Evaluation per prediction.

        """

        if model_names is None:
            model_names = [f"Model_{i}" for i in range(len(model_predictions))]
        else:
            assert_that(model_names).is_length(len(model_predictions))

        model_evaluations = [
            self.evaluator.evaluate(
                ground_truth,
                model_prediction=pred,
                model_name=model_name,
                sample_weights=sample_weights,
            )
            for pred, model_name in zip(model_predictions, model_names)
        ]

        return Comparison(model_evaluations)


def _get_and_supplement_model_names(model_evaluations: List[Evaluation]):
    return [
        evaluation.model_name
        if evaluation.model_name is not None
        else f"model_{i_model}"
        for i_model, evaluation in enumerate(model_evaluations)
    ]


def _model_evaluations_to_data_frame(
    model_evaluations: List[Evaluation],
) -> pd.DataFrame:
    quantity_names = [q.name for q in model_evaluations[0].quantities]

    # create one row per quantity
    data = []
    for i_q, quantity_name in enumerate(quantity_names):
        row = [quantity_name]
        for evaluation in model_evaluations:
            quantity = evaluation.quantities[i_q]
            assert_that(quantity.name).is_equal_to(quantity_name)
            row.append(quantity.value)
        data.append(row)

    model_names = _get_and_supplement_model_names(model_evaluations)
    return pd.DataFrame(data, columns=["Quantity"] + model_names)


def _check_consistency(model_evaluations: List[Evaluation]):
    if len(model_evaluations) == 0:
        return

    first = model_evaluations[0]
    for evaluation in model_evaluations:
        assert_that(evaluation.primary_metric).is_equal_to(first.primary_metric)
        assert_that(len(evaluation.quantities)).is_equal_to(len(first.quantities))
        for q, q_first in zip(evaluation.quantities, first.quantities):
            # check that everything except the value is equal
            assert_that(q.name).is_equal_to(q_first.name)
            assert_that(q.higher_is_better).is_equal_to(q_first.higher_is_better)
            assert_that(q.description).is_equal_to(q_first.description)

    not_none_model_names = [
        ms.model_name for ms in model_evaluations if ms.model_name is not None
    ]
    assert_that(not_none_model_names).does_not_contain_duplicates()


good_color = "#b2ffb2"


def _highlight_max(data):
    """Highlights the maximum in a Series or DataFrame.
    Checkout http://pandas.pydata.org/pandas-docs/stable/style.html for cool stuff.
    """
    attr = "background-color: {}".format(good_color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        # noinspection PyTypeChecker
        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )


def _highlight_min(data):
    """Highlights the minimum in a Series or DataFrame.
    Checkout http://pandas.pydata.org/pandas-docs/stable/style.html for cool stuff.
    """
    attr = "background-color: {}".format(good_color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        # noinspection PyTypeChecker
        return [attr if v else "" for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(
            np.where(is_min, attr, ""), index=data.index, columns=data.columns
        )


def _html_comparison_table(
    model_evaluations: List[Evaluation], include_spacer: bool
) -> str:
    _check_consistency(model_evaluations)
    primary_metric = model_evaluations[0].primary_metric
    n_models = len(model_evaluations)

    scores_data_frame = _model_evaluations_to_data_frame(
        [
            evaluation.filtered(keep_higher_is_better=True)
            for evaluation in model_evaluations
        ]
    )

    losses_data_frame = _model_evaluations_to_data_frame(
        [
            evaluation.filtered(keep_lower_is_better=True)
            for evaluation in model_evaluations
        ]
    )

    neutral_data_frame = _model_evaluations_to_data_frame(
        [
            evaluation.filtered(keep_neutral_quantities=True)
            for evaluation in model_evaluations
        ]
    )

    def is_primary_metric(a_metric: str):
        return a_metric.lower() == primary_metric

    def highlight_primary_metric(data):
        attr = "font-weight: bold; font-size: 120%;"
        if data.ndim == 1:
            metric = data[0].lower()
            if is_primary_metric(metric):
                return [attr for v in data]
            else:
                return ["" for v in data]
        else:  # from .apply(axis=None)
            good_things = np.ones_like(data).astype(bool)
            return pd.DataFrame(
                np.where(good_things, "", ""), index=data.index, columns=data.columns
            )

    def stylish_table_html(df: pd.DataFrame, highlight_fn=None) -> str:
        df_styled = df.style.set_properties(width="400px").format(_format_numbers)
        df_styled = df_styled.apply(highlight_primary_metric, axis=1)
        if highlight_fn is None:
            return df_styled.render()
        else:
            return df_styled.apply(highlight_fn, axis=1, subset=df.columns[1:]).render()

    html_output = ""

    # increase usable Jupyter notebook width when comparing many models
    if n_models > 3:
        html_output += "<style>.container { width:90% !important; }</style>"

    if len(scores_data_frame):
        html_output += "<h2>Scores (higher is better)</h2>"
        html_output += stylish_table_html(
            scores_data_frame, _highlight_max if n_models > 1 else None
        )

    if len(losses_data_frame):
        html_output += "<h2>Losses (lower is better)</h2>"
        html_output += stylish_table_html(
            losses_data_frame, _highlight_min if n_models > 1 else None
        )

    if len(neutral_data_frame):
        html_output += "<h2>Other Quantities</h2>"
        html_output += stylish_table_html(neutral_data_frame)

    # hide DataFrame indices
    # noinspection PyTypeChecker
    html_output += """
            <style>
            .row_heading {
                display: none;
            }
            .blank.level0 {
                display: none;
            }
            </style>
            """

    html_output += "<br><br>"

    # TODO check figure consistency

    # show rows of figures
    rows = []
    for i_figure, _ in enumerate(model_evaluations[0].lazy_figures):
        row_of_figures = [
            evaluation.lazy_figures[i_figure]()
            for i_model, evaluation in enumerate(model_evaluations)
        ]
        if include_spacer:
            row_of_figures = [Spacer()] + row_of_figures
        rows.append(bokeh.layouts.row(row_of_figures, sizing_mode="scale_width"),)

    html_output += file_html(
        models=column(rows, sizing_mode="scale_width"),
        resources=CDN,
        title="Comparison",
    )

    return html_output


def _format_numbers(entry):
    try:
        flt = float(entry)
        return "{:.3f}".format(flt)
    except ValueError:
        return entry
