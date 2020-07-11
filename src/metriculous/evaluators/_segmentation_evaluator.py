from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from bokeh.plotting import Figure
from sklearn import metrics as sklmetrics

from .._evaluation import Evaluation, Evaluator, Quantity
from ..evaluators._classification_figures_bokeh import _bokeh_output_histogram
from ..evaluators._segmentation_figures_bokeh import _bokeh_heatmap


class SegmentationEvaluator(Evaluator):

    """
    Implementation of the Segmentation Evaluator which should work well for most
    image segmentation problems.

    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        class_weights: Optional[Sequence[float]] = None,
        filter_quantities: Optional[Callable[[str], bool]] = None,
        filter_figures: Optional[Callable[[str], bool]] = None,
        primary_metric: Optional[str] = None,
    ):
        """
        Initializes the segmentation evaluator

        Args:
            num_classes:
                The number of classes
            class_names:
                Optional, names of classes
            class_weights:
                Optional, weights of classes in the same order as class_names. These
                weights don't necessarily need to add up to 1.0 as the weights are
                normalized but their ratios should reflect the weight distribution
                desired.
            filter_quantities:
                Callable that receives a quantity name and returns `False` if the
                quantity should be excluded.
                Examples:
                    `filter_quantities=lambda name: "vs Rest" not in name`
                    `filter_quantities=lambda name: "ROC" in name`
            filter_figures:
                Callable that receives a figure title and returns `False` if the figure
                should be excluded.
                Examples:
                    `filter_figures=lambda name: "vs Rest" not in name`
                    `filter_figures=lambda name: "ROC" in name`
            primary_metric:
                Optional string to specify the most important metric that should be used
                for model selection.

        """

        self.num_classes = num_classes

        if class_names is None:
            self.class_names: Sequence[str] = [
                "class_{}".format(i) for i in range(num_classes)
            ]
        else:
            self.class_names = class_names

        if class_weights is None:
            self.class_weights = [1.0 / num_classes] * num_classes
        else:
            total = sum(class_weights)
            self.class_weights = [weight / total for weight in class_weights]

        self.filter_quantities = (
            (lambda name: True) if filter_quantities is None else filter_quantities
        )
        self.filter_figures = (
            (lambda name: True) if filter_figures is None else filter_figures
        )

        self.primary_metric = primary_metric

        # Check for shape consistency

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "The number of classes doesn't match the number of the class names"
            )

        if len(self.class_weights) != self.num_classes:
            raise ValueError(
                "The number of classes doesn't match the number of the class weights"
            )

    def evaluate(
        self,
        ground_truth: np.ndarray,
        model_prediction: np.ndarray,
        model_name: str,
        sample_weights: Optional[Iterable[float]] = None,
    ) -> Evaluation:

        """

        Args:
             ground_truth:
                A 3D array of the shape - (Num_Samples, Height, Width)
             model_prediction:
                A 3D array with the same shape as ground_truth with each channel
                being the prediction of the model for the corresponding image.
             model_name:
                Name of the model to be evaluated
             sample_weights:
                Sequence of floats to modify the influence of individual samples on the
                statistics that will be measured.

        Returns:
            An Evaluation object containing Quantities and Figures that are useful for
            most segmentation problems.

        """

        if sample_weights is not None:
            raise NotImplementedError(
                "SegmentationEvaluator currently doesn't support sample weights"
            )

        if ground_truth.shape != model_prediction.shape:
            raise ValueError(
                (
                    f"The shape of the ground truth and the model predictions should be"
                    f"the same. Got ground_truth_shape: {ground_truth.shape}, "
                    f"model_predictions.shape: {model_prediction.shape}"
                )
            )

        if ground_truth.ndim != 3:
            raise ValueError(
                f"Ground Truth must be a 3D array. Got an {ground_truth.ndim}-d array"
            )

        if model_prediction.ndim != 3:
            raise ValueError(
                (
                    f"Model prediction must be a 3D array. "
                    f"Got a {model_prediction.ndim}-d array"
                )
            )

        quantities = [
            q
            for q in self._quantities(model_prediction, ground_truth)
            if self.filter_quantities(q.name)
        ]

        unfiltered_lazy_figures = self._lazy_figures(
            model_name, model_prediction, ground_truth
        )

        return Evaluation(
            quantities=quantities,
            lazy_figures=[
                function
                for name, function in unfiltered_lazy_figures
                if self.filter_figures(name)
            ],
            model_name=model_name,
            primary_metric=self.primary_metric,
        )

    def _lazy_figures(
        self, model_name: str, y_pred: np.ndarray, y_true: np.ndarray
    ) -> List[Tuple[str, Callable[[], Figure]]]:

        lazy_figures = []

        class_distribution_figure_name = "Class Distribution"

        def class_distribution_figure() -> Figure:
            figure = _bokeh_output_histogram(
                y_true=y_true,
                y_pred=y_pred,
                class_names=self.class_names,
                title_rows=[model_name, class_distribution_figure_name],
                sample_weights=None,
            )
            figure.yaxis.axis_label = "Number of Pixels"
            return figure

        lazy_figures.append((class_distribution_figure_name, class_distribution_figure))

        for class_label, class_name in enumerate(self.class_names):
            lazy_figures.append(
                (
                    f"Heatmap for {class_name}",
                    lambda: _bokeh_heatmap(
                        y_true=y_true,
                        y_pred=y_pred,
                        class_label=class_label,
                        class_name=class_name,
                    ),
                )
            )

        return lazy_figures

    def _quantities(self, y_pred: np.ndarray, y_true: np.ndarray) -> Sequence[Quantity]:

        # Flattened them as jaccard_score requires it in this way
        y_true_flattened, y_pred_flattened = y_true.flatten(), y_pred.flatten()

        quantities = list()
        weighted_miou = 0.0

        class_specific_miou = sklmetrics.jaccard_score(
            y_true_flattened, y_pred_flattened, average=None
        )

        if len(class_specific_miou) != self.num_classes:
            raise ValueError(
                (
                    f"The number of classes specified ({self.num_classes}) doesn't "
                    f"match with the number of classes actually present in the "
                    f"ground_truth/predictions ({len(class_specific_miou)}). Update the"
                    f" num_classes, class_names & class_weights parameters accordingly"
                )
            )
        for class_name, class_weight, value in zip(
            self.class_names, self.class_weights, class_specific_miou
        ):

            quantities.append(
                Quantity(f"{class_name} mIoU", value, higher_is_better=True)
            )

            weighted_miou += class_weight * value

        quantities.append(
            Quantity("Class weighted Mean mIoU", weighted_miou, higher_is_better=True)
        )

        return quantities
