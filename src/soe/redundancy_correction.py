"""Reduce viewpoint sets using aggregate OPF contribution.

This first version deliberately DOES NOT account for overlap between viewpoint
cutouts. Each cutout is collapsed to one aggregate score, viewpoints are ranked
by that score, and the smallest ranked prefix that reaches the requested
retention fraction is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .viewpoint import ViewpointOPFResult


@dataclass(frozen=True, slots=True)
class ViewpointScore:
    """Aggregate information score for one viewpoint."""

    identifier: str
    score: float
    fraction: float
    cumulative_fraction: float


@dataclass(frozen=True, slots=True)
class RedundancyCorrectionResult:
    """Result of reducing a set of viewpoint OPF cutouts."""

    selected: tuple[ViewpointScore, ...]
    discarded: tuple[ViewpointScore, ...]
    score_vector: np.ndarray
    total_score: float


def aggregate_viewpoint_score(
    viewpoint: ViewpointOPFResult,
) -> float:
    """Collapse one viewpoint OPF cutout into one aggregate score.

    Only valid, finite, positive OPF values contribute. Negative OPF values
    represent no useful observational contribution for this first version.
    """

    field = np.asarray(
        viewpoint.field,
        dtype=np.float64,
    )

    valid_mask = np.asarray(
        viewpoint.valid_mask,
        dtype=bool,
    )

    if field.shape != valid_mask.shape:
        raise ValueError(
            "The viewpoint OPF field and valid mask must have the same shape."
        )

    values = field[
        valid_mask & np.isfinite(field)
    ]

    if values.size == 0:
        return 0.0

    # We are measuring useful information, not allowing poor/negative areas to
    # cancel useful areas elsewhere in the cutout.
    useful_values = np.clip(
        values,
        0.0,
        None,
    )

    return float(np.sum(useful_values))


def build_score_vector(
    viewpoints: list[ViewpointOPFResult],
) -> np.ndarray:
    """Return one aggregate OPF score for every viewpoint.

    Position ``i`` in the output corresponds to ``viewpoints[i]``.
    """

    return np.asarray(
        [
            aggregate_viewpoint_score(viewpoint)
            for viewpoint in viewpoints
        ],
        dtype=np.float64,
    )


def correct_redundancy(
    viewpoints: list[ViewpointOPFResult],
    *,
    retention: float = 0.95,
) -> RedundancyCorrectionResult:
    """Select the smallest highest-scoring subset meeting ``retention``.

    ``retention=0.95`` means: rank the aggregate viewpoint scores from highest
    to lowest, then keep the shortest prefix whose cumulative score accounts
    for at least 95% of the summed candidate-viewpoint score.

    This is score-concentration pruning, not yet overlap-aware marginal-gain
    pruning.
    """

    if not 0.0 < retention <= 1.0:
        raise ValueError(
            "retention must be greater than 0 and at most 1."
        )

    if not viewpoints:
        return RedundancyCorrectionResult(
            selected=(),
            discarded=(),
            score_vector=np.array([], dtype=np.float64),
            total_score=0.0,
        )

    score_vector = build_score_vector(viewpoints)
    total_score = float(np.sum(score_vector))

    if total_score <= 0.0:
        scores = tuple(
            ViewpointScore(
                identifier=viewpoint.viewpoint.identifier,
                score=0.0,
                fraction=0.0,
                cumulative_fraction=0.0,
            )
            for viewpoint in viewpoints
        )

        return RedundancyCorrectionResult(
            selected=(),
            discarded=scores,
            score_vector=score_vector,
            total_score=0.0,
        )

    # Highest-value viewpoints first.
    order = np.argsort(score_vector)[::-1]

    ranked: list[ViewpointScore] = []
    cumulative_score = 0.0

    for index in order:
        score = float(score_vector[index])
        fraction = score / total_score

        cumulative_score += score
        cumulative_fraction = cumulative_score / total_score

        ranked.append(
            ViewpointScore(
                identifier=viewpoints[index].viewpoint.identifier,
                score=score,
                fraction=fraction,
                cumulative_fraction=cumulative_fraction,
            )
        )

    # Find the smallest prefix that reaches the required retention.
    selected_count = len(ranked)

    for i, result in enumerate(ranked):
        if result.cumulative_fraction >= retention:
            selected_count = i + 1
            break

    return RedundancyCorrectionResult(
        selected=tuple(ranked[:selected_count]),
        discarded=tuple(ranked[selected_count:]),
        score_vector=score_vector,
        total_score=total_score,
    )


def select_viewpoints_from_scores(
    scores: np.ndarray,
    retention: float = 0.95,
) -> tuple[list[int], np.ndarray]:
    """Small standalone helper for testing the ranking logic with fake scores."""

    scores = np.asarray(scores, dtype=float)

    if not 0.0 < retention <= 1.0:
        raise ValueError(
            "retention must be greater than 0 and at most 1."
        )

    if scores.size == 0:
        return [], np.array([], dtype=int)

    if np.any(~np.isfinite(scores)):
        raise ValueError("Scores must contain only finite values.")

    if np.any(scores < 0):
        raise ValueError("Scores must not be negative.")

    total_score = float(np.sum(scores))

    if total_score <= 0.0:
        return [], np.argsort(scores)[::-1]

    fractions = scores / total_score
    order = np.argsort(scores)[::-1]
    ranked_fractions = fractions[order]
    cumulative_fractions = np.cumsum(ranked_fractions)

    selected: list[int] = []

    for rank, index in enumerate(order):
        selected.append(int(index))

        if cumulative_fractions[rank] >= retention:
            break

    return selected, order


