from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.data.concept import Concept
from pyclad.metrics.base.base_metric import BaseMetric
from pyclad.metrics.continual.concepts_metric import (
    ConceptLevelMatrix,
    ScheduleAwareMetric,
    StepwiseConceptMetric,
    SummarizedMetric,
)
from pyclad.output.output_writer import InfoProvider


class ConceptMetricCallback(Callback, InfoProvider):
    """Collects a base metric for every (training step, evaluated concept) pair.

    Rows are training steps in the order they were learned, columns are evaluated concepts
    in the order they were first evaluated. In the default setting both axes hold the same
    concept names and the matrix is square (``N x N``).

    When training steps group several categories (see :mod:`pyclad.data.grouping`) the rows
    are the grouped steps while the columns remain the original categories, so the matrix
    becomes rectangular (``T x N``) and the axes no longer share names. Metrics that need to
    know when each category entered training are passed via ``schedule_aware_metrics``
    together with ``first_seen_step``.

    :param schedule_aware_metrics: metrics computed as ``compute(matrix, first_seen_steps)``.
    :param first_seen_step: category name -> index of the training step that first includes
        it, typically ``StepScheduledConceptsDataset.first_seen_step()``. Required whenever
        ``schedule_aware_metrics`` is non-empty.
    """

    def __init__(
        self,
        base_metric: BaseMetric,
        summarized_metrics: Iterable[SummarizedMetric],
        stepwise_metrics: Iterable[StepwiseConceptMetric] = None,
        schedule_aware_metrics: Iterable[ScheduleAwareMetric] = (),
        first_seen_step: Optional[Mapping[str, int]] = None,
    ):
        self._base_metric: BaseMetric = base_metric
        self._metric_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._learned_concepts: List[str] = []
        self._evaluated_concepts: List[str] = []
        self._summarized_metrics = summarized_metrics
        self._stepwise_metrics = stepwise_metrics if stepwise_metrics is not None else []
        self._schedule_aware_metrics: List[ScheduleAwareMetric] = list(schedule_aware_metrics)
        self._first_seen_step = dict(first_seen_step) if first_seen_step is not None else None

        validate_schedule_aware_configuration(self._schedule_aware_metrics, self._first_seen_step)

    def after_training(self, learned_concept: Concept):
        self._learned_concepts.append(learned_concept.name)

    def after_evaluation(
        self,
        evaluated_concept: Concept,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_scores: np.ndarray,
        *args,
        **kwargs,
    ):
        assert (
            evaluated_concept.name not in self._metric_matrix[self._learned_concepts[-1]]
        ), "The same concept should not be evaluated twice after the same learned concept"

        metric_value = self._base_metric.compute(anomaly_scores=anomaly_scores, y_true=y_true, y_pred=y_pred)
        self._metric_matrix[self._learned_concepts[-1]][evaluated_concept.name] = metric_value

        if evaluated_concept.name not in self._evaluated_concepts:
            self._evaluated_concepts.append(evaluated_concept.name)

    def info(self) -> Dict[str, Any]:
        concept_level_matrix = build_dense_matrix(self._metric_matrix, self._learned_concepts, self._evaluated_concepts)
        summarized_metrics = {m.name(): m.compute(concept_level_matrix) for m in self._summarized_metrics}
        stepwise_metrics = {m.name(): m.compute(concept_level_matrix) for m in self._stepwise_metrics}

        payload = {
            "base_metric_name": self._base_metric.name(),
            "metrics": summarized_metrics,
            "stepwise_metrics": stepwise_metrics,
            "concepts_order": self._learned_concepts,
            "test_order": self._evaluated_concepts,
            "metric_matrix": self._metric_matrix,
        }

        if self._first_seen_step is not None:
            first_seen_steps = align_first_seen_steps(self._first_seen_step, self._evaluated_concepts)
            payload["first_seen_step"] = dict(zip(self._evaluated_concepts, first_seen_steps))
            payload["schedule_aware_metrics"] = {
                m.name(): m.compute(concept_level_matrix, first_seen_steps) for m in self._schedule_aware_metrics
            }

        return {f"concept_metric_callback_{self._base_metric.name()}": payload}


def build_dense_matrix(
    metric_matrix: Mapping[str, Mapping[str, float]],
    train_order: Sequence[str],
    test_order: Sequence[str],
) -> ConceptLevelMatrix:
    """Turn the sparse ``{learned: {evaluated: value}}`` mapping into a dense matrix.

    Rows follow ``train_order``, columns follow ``test_order``. The scenario evaluates every
    test concept after every training step, so a missing cell means a broken run rather than
    a legitimately unknown value.

    :raises ValueError: when any (training step, evaluated concept) pair is missing.
    """
    if len(train_order) == 0 or len(test_order) == 0:
        return [[]]

    values: ConceptLevelMatrix = []
    for learned_concept in train_order:
        row = []
        for evaluated_concept in test_order:
            if evaluated_concept not in metric_matrix.get(learned_concept, {}):
                raise ValueError(
                    f"Concept {evaluated_concept!r} was not evaluated after training step "
                    f"{learned_concept!r}; the concept-level matrix is incomplete."
                )
            row.append(metric_matrix[learned_concept][evaluated_concept])
        values.append(row)
    return values


def align_first_seen_steps(first_seen_step: Mapping[str, int], test_order: Sequence[str]) -> List[int]:
    """Order a ``{category: first seen step}`` mapping to match the matrix columns.

    :raises ValueError: when an evaluated concept is absent from the mapping.
    """
    missing = [name for name in test_order if name not in first_seen_step]
    if missing:
        raise ValueError(
            f"first_seen_step is missing an entry for evaluated concepts {missing}. "
            "It must cover every evaluated concept; build it with "
            "pyclad.data.grouping.compute_first_seen_step."
        )
    return [int(first_seen_step[name]) for name in test_order]


def validate_schedule_aware_configuration(
    schedule_aware_metrics: Sequence[ScheduleAwareMetric], first_seen_step: Optional[Mapping[str, int]]
) -> None:
    """:raises ValueError: when schedule-aware metrics are requested without a mapping to feed them."""
    if schedule_aware_metrics and first_seen_step is None:
        raise ValueError(
            "schedule_aware_metrics require first_seen_step, mapping every evaluated concept "
            "to the training step that first includes it. Use "
            "StepScheduledConceptsDataset.first_seen_step()."
        )
