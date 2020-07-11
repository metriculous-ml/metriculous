"""
This module contains data types and interfaces that are used throughout the library.
Here we do not make any assumptions about the structure of ground truth and predictions.
"""
from dataclasses import dataclass, field, replace
from typing import Callable, Generic, Optional, Sequence, TypeVar, Union

from bokeh.plotting import Figure


@dataclass(frozen=True)
class Quantity:
    name: str
    value: Union[float, str]
    higher_is_better: Optional[bool] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class Evaluation:
    model_name: str
    quantities: Sequence[Quantity] = field(default_factory=list)
    lazy_figures: Sequence[Callable[[], Figure]] = field(default_factory=list)
    primary_metric: Optional[str] = None

    def get_by_name(self, quantity_name: str) -> Quantity:
        # Number of quantities is usually small,
        # so do not bother with internal dict for lookup
        for q in self.quantities:
            if quantity_name == q.name:
                return q
        raise ValueError(f"Could not find quantity named {quantity_name}")

    def get_primary(self) -> Optional[Quantity]:
        if self.primary_metric is None:
            return None
        return self.get_by_name(self.primary_metric)

    def figures(self) -> Sequence[Figure]:
        return [f() for f in self.lazy_figures]

    def filtered(
        self,
        keep_higher_is_better: bool = False,
        keep_lower_is_better: bool = False,
        keep_neutral_quantities: bool = False,
    ) -> "Evaluation":
        return replace(
            self,
            quantities=[
                q
                for q in self.quantities
                if any(
                    [
                        (q.higher_is_better is True and keep_higher_is_better),
                        (q.higher_is_better is False and keep_lower_is_better),
                        (q.higher_is_better is None and keep_neutral_quantities),
                    ]
                )
            ],
        )


G = TypeVar("G", contravariant=True)
P = TypeVar("P", contravariant=True)


class Evaluator(Generic[G, P]):
    """
    Interface to be implemented by the user to compute quantities and charts that are
    relevant and applicable to the problem at hand.
    """

    def evaluate(
        self,
        ground_truth: G,
        model_prediction: P,
        model_name: str,
        sample_weights: Optional[Sequence[float]] = None,
    ) -> Evaluation:
        """Generates an Evaluation from ground truth and a model prediction."""
        raise NotImplementedError
