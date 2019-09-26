from typing import Callable
from typing import Optional
from typing import Sequence

import numpy as np
from assertpy import assert_that
from scipy.stats import entropy
from sklearn import metrics as sklmetrics

from ._classification_utils import check_normalization
from .._evaluation import Evaluation
from .._evaluation import Evaluator
from .._evaluation import Quantity
from ..evaluators._classification_figures_bokeh import _bokeh_automation_rate_analysis
from ..evaluators._classification_figures_bokeh import _bokeh_confusion_matrix
from ..evaluators._classification_figures_bokeh import _bokeh_confusion_scatter
from ..evaluators._classification_figures_bokeh import _bokeh_output_histogram
from ..evaluators._classification_figures_bokeh import _bokeh_precision_recall_curve
from ..evaluators._classification_figures_bokeh import _bokeh_roc_curve
from ..metrics import top_n_accuracy
from ..utilities import sample_weights_simulating_class_distribution


class ClassificationEvaluator(Evaluator):
    """
    Default Evaluator implementation that serves well for most classification problems.

    """

    def __init__(
        self,
        class_names: Optional[Sequence[str]] = None,
        one_vs_all_quantities=True,
        one_vs_all_figures=False,
        top_n_accuracies: Sequence[int] = (),
        filter_quantities: Optional[Callable[[str], bool]] = None,
        filter_figures: Optional[Callable[[str], bool]] = None,
        primary_metric: Optional[str] = None,
        simulated_class_distribution: Optional[Sequence[float]] = None,
        class_label_rotation_x="horizontal",
        class_label_rotation_y="vertical",
    ):
        """
        Initializes the evaluator with the option to overwrite the default settings.

        Args:
            class_names:
                Optional, names of the classes.
            one_vs_all_quantities:
                If `True` show quantities like "ROC AUC Class_i vs Rest" for all i.
            one_vs_all_figures:
                If `True` show figures like "ROC Curve Class_i vs Rest" for all i.
            top_n_accuracies:
                A sequence of positive integers to specify which top-N accuracy metrics
                should be computed.
                Example: `top_n_accuracies=[2, 3, 5, 10]`
            filter_quantities:
                Callable that receives a quantity name and returns `False` if the
                quantity should be excluded.
                Example: `filter_quantities=lambda name: "vs Rest" not in name`
            filter_figures:
                Callable that receives a figure title and returns `False` if the figure
                should be excluded.
                Example: `filter_figures=lambda name: "ROC" in name`
            primary_metric:
                Optional string to specify the most important metric that should be used
                for model selection.
            simulated_class_distribution:
                Optional sequence of floats that indicates a hypothetical class
                distribution on which models should be evaluated. If not `None`, sample
                weights will be computed and used to simulate the desired class
                distribution.
            class_label_rotation_x:
                Rotation of x-axis tick labels for figures with class name tick labels.
            class_label_rotation_y:
                Rotation of y-axis tick labels for figures with class name tick labels.

        """
        self.class_names = class_names
        self.one_vs_all_quantities = one_vs_all_quantities
        self.one_vs_all_figures = one_vs_all_figures

        self.top_n_accuracies = top_n_accuracies
        assert all(isinstance(val, int) for val in self.top_n_accuracies)
        assert all(val >= 1 for val in self.top_n_accuracies)

        self.filter_quantities = (
            (lambda name: True) if filter_quantities is None else filter_quantities
        )
        self.filter_figures = (
            (lambda name: True) if filter_figures is None else filter_figures
        )
        self.primary_metric = primary_metric

        if simulated_class_distribution is not None:
            check_normalization(simulated_class_distribution, axis=0)
            np.testing.assert_equal(
                np.asarray(simulated_class_distribution) > 0.0, True
            )

        self.simulated_class_distribution = simulated_class_distribution

        self.class_label_rotation_x = class_label_rotation_x
        self.class_label_rotation_y = class_label_rotation_y

    def evaluate(
        self,
        ground_truth: np.ndarray,
        model_prediction: np.ndarray,
        model_name: str,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Evaluation:
        """
        Computes Quantities and generates Figures that are useful for most
        classification problems.

        Args:
            model_prediction:
                Sequence of 2d arrays where each array corresponds to a model
                and each row is a probability distribution.
            ground_truth:
                2d array with each row being a probability distribution.
            model_name:
                Name of the model that is being evaluated.
            sample_weights:
                Sequence of floats to modify the influence of individual samples on the
                statistics that will be measured.

        Returns:
            An Evaluation object containing Quantities and Figures that are useful for
            most classification problems.

        """

        # === Preparations =============================================================
        # give variables more specific names
        y_pred_proba = model_prediction
        y_true_proba = ground_truth
        # and delete interface parameter names to avoid confusion
        del model_prediction
        del ground_truth

        n_classes = y_true_proba.shape[1]

        if self.class_names is None:
            self.class_names = ["class_{}".format(i) for i in range(n_classes)]
        assert len(self.class_names) == y_true_proba.shape[1]

        # check shapes
        assert np.ndim(y_true_proba) == 2
        assert (
            y_true_proba.shape == y_pred_proba.shape
        ), f"{y_true_proba.shape} != {y_pred_proba.shape}"

        # check normalization
        check_normalization(y_true_proba, axis=1)
        check_normalization(y_pred_proba, axis=1)
        np.testing.assert_equal(y_true_proba >= 0.0, True)
        np.testing.assert_equal(y_pred_proba >= 0.0, True)

        # compute non-probabilistic class decisions
        y_true = np.argmax(y_true_proba, axis=1)
        y_pred = np.argmax(y_pred_proba, axis=1)

        # make one-hot arrays, which are required for some sklearn metrics
        y_true_one_hot: np.ndarray = np.eye(n_classes)[y_true]
        y_pred_one_hot: np.ndarray = np.eye(n_classes)[y_pred]

        # process sample_weights parameter and self.simulated_class_distribution
        if sample_weights is not None:
            assert self.simulated_class_distribution is None, (
                "Cannot use `sample_weights` with ClassificationEvaluator that was "
                "initialized with `simulated_class_distribution`."
            )
            sample_weights = np.asarray(sample_weights)
            assert_that(sample_weights.ndim).is_equal_to(1)
            assert_that(sample_weights.shape).is_equal_to((len(y_pred),))
            np.testing.assert_array_equal(sample_weights >= 0.0, True)
        elif self.simulated_class_distribution is not None:
            assert_that(np.shape(self.simulated_class_distribution)).is_equal_to(
                (n_classes,)
            )
            sample_weights = sample_weights_simulating_class_distribution(
                y_true=y_true,
                hypothetical_class_distribution=self.simulated_class_distribution,
            )

        # === Quantities ===============================================================
        # Note: Optimization potential here for problems with many classes.
        # We are currently computing all quantities and then throwing away some of them,
        # rather than only computing those that are requested by self.filter_quantities
        quantities = [
            q
            for q in self._quantities(
                y_pred,
                y_pred_one_hot,
                y_pred_proba,
                y_true,
                y_true_one_hot,
                y_true_proba,
                maybe_sample_weights=sample_weights,
            )
            if self.filter_quantities(q.name)
        ]

        # === Figures ==================================================================
        figures = self._figures(
            model_name,
            y_pred=y_pred,
            y_pred_one_hot=y_pred_one_hot,
            y_pred_proba=y_pred_proba,
            y_true=y_true,
            y_true_one_hot=y_true_one_hot,
            y_true_proba=y_true_proba,
            maybe_sample_weights=sample_weights,
        )

        return Evaluation(
            quantities=quantities,
            figures=figures,
            model_name=model_name,
            primary_metric=self.primary_metric,
        )

    def _figures(
        self,
        model_name: str,
        y_pred: np.ndarray,
        y_pred_one_hot: np.ndarray,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        y_true_one_hot: np.ndarray,
        y_true_proba: np.ndarray,
        maybe_sample_weights: Optional[np.ndarray],
    ):
        figures = []

        # --- Histogram of predicted and ground truth classes ---
        if maybe_sample_weights is None:
            figure_name = "Class Distribution"
            if self.filter_figures(figure_name):
                figures.append(
                    _bokeh_output_histogram(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_names=self.class_names,
                        title_rows=[model_name, figure_name],
                        sample_weights=None,
                        x_label_rotation=self.class_label_rotation_x,
                    )
                )
        else:
            figure_name = "Unweighted Class Distribution"
            if self.filter_figures(figure_name):
                figures.append(
                    _bokeh_output_histogram(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_names=self.class_names,
                        title_rows=[model_name, figure_name],
                        sample_weights=None,
                        x_label_rotation=self.class_label_rotation_x,
                    )
                )

            figure_name = "Weighted Class Distribution"
            if self.filter_figures(figure_name):
                figures.append(
                    _bokeh_output_histogram(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_names=self.class_names,
                        title_rows=[model_name, figure_name],
                        sample_weights=maybe_sample_weights,
                        x_label_rotation=self.class_label_rotation_x,
                    )
                )

        # --- Confusion Scatter Plot ---
        figure_name = "Confusion Scatter Plot"
        if maybe_sample_weights is None and self.filter_figures(figure_name):
            figures.append(
                _bokeh_confusion_scatter(
                    y_true=y_true,
                    y_pred=y_pred,
                    class_names=self.class_names,
                    title_rows=[model_name, figure_name],
                    x_label_rotation=self.class_label_rotation_x,
                    y_label_rotation=self.class_label_rotation_y,
                )
            )

        # --- Confusion Matrix ---
        figure_name = "Confusion Matrix"
        if maybe_sample_weights is None and self.filter_figures(figure_name):
            figures.append(
                _bokeh_confusion_matrix(
                    y_true=y_true,
                    y_pred=y_pred,
                    class_names=self.class_names,
                    title_rows=[model_name, figure_name],
                    x_label_rotation=self.class_label_rotation_x,
                    y_label_rotation=self.class_label_rotation_y,
                )
            )

        # --- Automation Rate Analysis ---
        figure_name = "Automation Rate Analysis"
        if self.filter_figures(figure_name):
            figures.append(
                _bokeh_automation_rate_analysis(
                    y_target_one_hot=y_true_one_hot,
                    y_pred_proba=y_pred_proba,
                    title_rows=[model_name, figure_name],
                    sample_weights=maybe_sample_weights,
                )
            )

        # --- ROC curves ---
        if self.one_vs_all_figures:
            for class_index, class_name in enumerate(self.class_names):
                figure_name = f"ROC {class_name} vs Rest"
                if self.filter_figures(figure_name):
                    figures.append(
                        _bokeh_roc_curve(
                            y_true_binary=(y_true == class_index),
                            y_pred_score=y_pred_proba[:, class_index],
                            title_rows=[model_name, figure_name],
                            sample_weights=maybe_sample_weights,
                        )
                    )

        # --- PR curves ---
        if self.one_vs_all_figures:
            for class_index, class_name in enumerate(self.class_names):
                figure_name = f"PR Curve {class_name} vs Rest"
                if self.filter_figures(figure_name):
                    figures.append(
                        _bokeh_precision_recall_curve(
                            y_true_binary=(y_true == class_index),
                            y_pred_score=y_pred_proba[:, class_index],
                            title_rows=[model_name, figure_name],
                            sample_weights=maybe_sample_weights,
                        )
                    )

        return figures

    def _quantities(
        self,
        y_pred: np.ndarray,
        y_pred_one_hot: np.ndarray,
        y_pred_proba: np.ndarray,
        y_true: np.ndarray,
        y_true_one_hot: np.ndarray,
        y_true_proba: np.ndarray,
        maybe_sample_weights: Optional[np.ndarray],
    ):
        quantities = []

        quantities.append(
            Quantity(
                "Accuracy",
                sklmetrics.accuracy_score(
                    y_true, y_pred, sample_weight=maybe_sample_weights
                ),
                higher_is_better=True,
            )
        )

        quantities.append(
            Quantity(
                "ROC AUC Macro Average",
                sklmetrics.roc_auc_score(
                    y_true_one_hot,
                    y_pred_proba,
                    average="macro",
                    sample_weight=maybe_sample_weights,
                ),
                higher_is_better=True,
            )
        )

        quantities.append(
            Quantity(
                "ROC AUC Micro Average",
                sklmetrics.roc_auc_score(
                    y_true_one_hot,
                    y_pred_proba,
                    average="micro",
                    sample_weight=maybe_sample_weights,
                ),
                higher_is_better=True,
            )
        )

        quantities.append(
            Quantity(
                "F1-Score Macro Average",
                sklmetrics.f1_score(
                    y_true_one_hot,
                    y_pred_one_hot,
                    average="macro",
                    sample_weight=maybe_sample_weights,
                ),
                higher_is_better=True,
            )
        )

        quantities.append(
            Quantity(
                "F1-Score Micro Average",
                sklmetrics.f1_score(
                    y_true_one_hot,
                    y_pred_one_hot,
                    average="micro",
                    sample_weight=maybe_sample_weights,
                ),
                higher_is_better=True,
            )
        )

        # --- Top-N accuracies ---
        for n in self.top_n_accuracies:
            quantities.append(
                Quantity(
                    f"Top-{n} Accuracy",
                    value=top_n_accuracy(
                        y_true, y_pred_proba, n=n, sample_weights=maybe_sample_weights
                    ),
                    higher_is_better=True,
                )
            )

        # --- One-vs-rest ROC AUC scores ---
        if self.one_vs_all_quantities:
            # noinspection PyTypeChecker
            roc_auc_scores: Sequence[float] = sklmetrics.roc_auc_score(
                y_true_one_hot,
                y_pred_proba,
                average=None,
                sample_weight=maybe_sample_weights,
            )
            for class_index, class_name in enumerate(self.class_names):
                quantities.append(
                    Quantity(
                        f"ROC AUC {class_name} vs Rest",
                        value=roc_auc_scores[class_index],
                        higher_is_better=True,
                    )
                )

        # --- One-vs-rest average precision scores ---
        if self.one_vs_all_quantities:
            # noinspection PyTypeChecker
            ap_scores: Sequence[float] = sklmetrics.average_precision_score(
                y_true_one_hot,
                y_pred_proba,
                average=None,
                sample_weight=maybe_sample_weights,
            )
            for class_index, class_name in enumerate(self.class_names):
                quantities.append(
                    Quantity(
                        f"Average Precision {class_name} vs Rest",
                        value=ap_scores[class_index],
                        higher_is_better=True,
                    )
                )

        # --- One-vs-rest F1-scores ---
        if self.one_vs_all_quantities:
            f1_scores = sklmetrics.f1_score(
                y_true_one_hot,
                y_pred_one_hot,
                average=None,
                sample_weight=maybe_sample_weights,
            )
            for class_index, class_name in enumerate(self.class_names):
                quantities.append(
                    Quantity(
                        f"F1-Score {class_name} vs Rest",
                        value=f1_scores[class_index],
                        higher_is_better=True,
                    )
                )

        # --- KL-divergence ---
        # keep in mind entropy(p, q) != entropy(q, p)
        kl_divergences = np.array(
            [
                entropy(pk=true_dist, qk=pred_dist)
                for true_dist, pred_dist in zip(y_true_proba, y_pred_proba)
            ]
        )
        quantities.append(
            Quantity(
                "Mean KLD(P=target||Q=prediction)",
                np.average(kl_divergences, weights=maybe_sample_weights),
                higher_is_better=False,
            )
        )

        # --- Log loss ---
        quantities.append(
            Quantity(
                "Log Loss",
                sklmetrics.log_loss(
                    y_true_one_hot, y_pred_one_hot, sample_weight=maybe_sample_weights
                ),
                higher_is_better=False,
            )
        )

        # --- Brier score loss ---
        # Be careful with sklmetrics.brier_score_loss, it deviates from Brier's
        # definition for multi-class problems.
        # See https://stats.stackexchange.com/questions
        # /403544/how-to-compute-the-brier-score-for-more-than-two-classes
        # and Wikipedia
        # noinspection PyTypeChecker
        quantities.append(
            Quantity(
                "Brier Score Loss",
                np.mean((y_pred_proba - y_true_one_hot) ** 2),
                higher_is_better=False,
            )
        )
        # noinspection PyTypeChecker
        quantities.append(
            Quantity(
                "Brier Score Loss (Soft Targets)",
                np.mean((y_pred_proba - y_true_proba) ** 2),
                higher_is_better=False,
            )
        )

        # --- entropy of prediction probability distributions ---
        entropies_pred = np.array([entropy(proba_dist) for proba_dist in y_pred_proba])
        quantities.append(Quantity("Max Entropy", entropies_pred.max()))
        quantities.append(
            Quantity(
                "Mean Entropy", np.average(entropies_pred, weights=maybe_sample_weights)
            )
        )
        quantities.append(Quantity("Min Entropy", entropies_pred.min()))
        quantities.append(Quantity("Max Probability", y_pred_proba.max()))
        quantities.append(Quantity("Min Probability", y_pred_proba.min()))
        return quantities
