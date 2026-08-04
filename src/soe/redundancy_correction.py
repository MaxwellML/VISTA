from __future__ import annotations
"""Reduce viewpoint sets using aggregate OPF contribution."""
"""DOES NOT CURRENTLY ACCOUNT FOR OVERLAP."""


from dataclasses import dataclass

import numpy as np



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
    """
    Collapse one viewpoint OPF cutout into one aggregate score.

    Only valid, positive OPF values contribute. Negative OPF values
    represent no useful observational contribution for this purpose.
    """

    field = np.asarray(
        viewpoint.field,
        dtype=np.float64,
    )

    valid_mask = np.asarray(
        viewpoint.valid_mask,
        dtype=bool,
    )

    values = field[
        valid_mask & np.isfinite(field)
    ]

    if values.size == 0:
        return 0.0

    # We are measuring useful information, not allowing poor/negative
    # areas to cancel useful areas elsewhere in the cutout.
    useful_values = np.clip(
        values,
        0.0,
        None,
    )

    return float(np.sum(useful_values))


def build_score_vector(
    viewpoints: list[ViewpointOPFResult],
) -> np.ndarray:
    """
    Return one aggregate OPF score for every viewpoint.

    Position i in the output corresponds to viewpoints[i].
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
    """
    Select the smallest number of highest-scoring viewpoints required
    to account for the requested fraction of aggregate viewpoint score.

    Example:
        retention=0.95

    means retain enough viewpoints to account for at least 95% of the
    total score represented by the candidate score vector.
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

        cumulative_fraction = (
            cumulative_score / total_score
        )

        ranked.append(
            ViewpointScore(
                identifier=(
                    viewpoints[index]
                    .viewpoint
                    .identifier
                ),
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
    retention: float = 0.95
):
    """
    Rank viewpoint scores from highest to lowest and select the
    smallest number required to reach the requested retention.
    """

    scores = np.asarray(scores, dtype=float)

    if scores.size == 0:
        return [], []

    if np.any(scores < 0):
        raise ValueError("Scores must not be negative.")

    total_score = np.sum(scores)

    if total_score == 0:
        return [], []

    # Fraction of total score contributed by each viewpoint
    fractions = scores / total_score

    # Sort highest score first
    order = np.argsort(scores)[::-1]

    ranked_scores = scores[order]
    ranked_fractions = fractions[order]

    cumulative_fractions = np.cumsum(ranked_fractions)

    selected = []

    for rank, index in enumerate(order):

        selected.append(index)

        if cumulative_fractions[rank] >= retention:
            break

    return selected, order


if __name__ == "__main__":

    # ---------------------------------------------------------
    # FAKE VIEWPOINT SCORES
    # ---------------------------------------------------------

    viewpoint_names = np.array([
        "VP_A",
        "VP_B",
        "VP_C",
        "VP_D",
        "VP_E",
        "VP_F",
        "VP_G",
        "VP_H",
    ])

    scores = np.array([
        82.0,   # VP_A
        12.0,   # VP_B
        47.0,   # VP_C
        5.0,    # VP_D
        31.0,   # VP_E
        3.0,    # VP_F
        18.0,   # VP_G
        2.0,    # VP_H
    ])

    retention = 0.90

    # ---------------------------------------------------------
    # RUN REDUNDANCY CORRECTOR
    # ---------------------------------------------------------

    selected, ranking = select_viewpoints_from_scores(
        scores,
        retention
    )

    total_score = np.sum(scores)

    print("\nVIEWPOINT RANKING")
    print("-----------------")

    cumulative = 0.0

    for rank, index in enumerate(ranking, start=1):

        fraction = scores[index] / total_score
        cumulative += fraction

        selected_marker = (
            "KEEP"
            if index in selected
            else "DISCARD"
        )

        print(
            f"{rank}. "
            f"{viewpoint_names[index]} | "
            f"score={scores[index]:.1f} | "
            f"fraction={fraction:.2%} | "
            f"cumulative={cumulative:.2%} | "
            f"{selected_marker}"
        )

    print("\nSelected viewpoints:")

    for index in selected:
        print(
            viewpoint_names[index],
            scores[index]
        )
        