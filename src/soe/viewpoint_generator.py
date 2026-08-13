"""Generate robust candidate viewpoints from an OPF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observability_potential_field import (
    ObservabilityPotentialFieldResult,
)
from .viewpoint import (
    ViewpointRegion,
    build_viewpoint_opf,
)


@dataclass(frozen=True, slots=True)
class ViewpointGenerationResult:
    """A generated viewpoint plus diagnostics explaining its selection."""

    viewpoint: ViewpointRegion
    peak_score: float
    local_mean: float
    local_std: float
    support_fraction: float
    neighbourhood_cells: int
    row: int
    column: int


@dataclass(frozen=True, slots=True)
class _NeighbourhoodStatistics:
    mean: float
    std: float
    support_fraction: float
    count: int


def generate_viewpoints_from_opf(
    opf_result: ObservabilityPotentialFieldResult,
    *,
    verification_radius_m: float,
    suppression_radius_m: float,
    max_viewpoints: int = 10,
    support_tolerance: float = 0.05,
    min_support_fraction: float = 0.25,
    min_neighbourhood_cells: int = 5,
    minimum_score: float | None = None,
) -> list[ViewpointGenerationResult]:
    """
    Generate robust viewpoints from high-valued OPF regions.

    Candidate cells are considered from highest to lowest OPF value.

    For each candidate:
      1. Create a ViewpointRegion centred on the candidate.
      2. Use build_viewpoint_opf() to extract its verification neighbourhood.
      3. Measure how much of that neighbourhood is close to the peak score.
      4. Reject isolated peaks with insufficient local support.
      5. Reject accepted viewpoints that would be too close together.
      6. Otherwise retain the existing ViewpointRegion plus diagnostics.

    support_tolerance defines how far below the peak a neighbouring
    cell may be while still supporting that peak.

    Example:
        peak_score = 0.90
        support_tolerance = 0.05

    A neighbouring cell supports the peak when:
        cell_score >= 0.85
    """

    _validate_parameters(
        opf_result=opf_result,
        verification_radius_m=verification_radius_m,
        suppression_radius_m=suppression_radius_m,
        max_viewpoints=max_viewpoints,
        support_tolerance=support_tolerance,
        min_support_fraction=min_support_fraction,
        min_neighbourhood_cells=min_neighbourhood_cells,
        minimum_score=minimum_score,
    )

    field = np.asarray(opf_result.field, dtype=np.float32)

    if field.ndim != 2:
        raise ValueError("OPF field must be two-dimensional.")

    valid_mask = (
        np.asarray(opf_result.valid_mask, dtype=bool)
        & np.isfinite(field)
    )

    if not np.any(valid_mask):
        return []

    candidate_rows, candidate_columns = np.nonzero(valid_mask)
    candidate_scores = field[
        candidate_rows,
        candidate_columns,
    ]

    if minimum_score is not None:
        keep = candidate_scores >= minimum_score
        candidate_rows = candidate_rows[keep]
        candidate_columns = candidate_columns[keep]
        candidate_scores = candidate_scores[keep]

    if candidate_scores.size == 0:
        return []

    # Highest OPF cells are examined first.
    order = np.argsort(candidate_scores)[::-1]
    candidate_rows = candidate_rows[order]
    candidate_columns = candidate_columns[order]
    candidate_scores = candidate_scores[order]

    selected: list[ViewpointGenerationResult] = []
    transform = opf_result.transform

    for row, column, peak_score in zip(
        candidate_rows,
        candidate_columns,
        candidate_scores,
    ): #through zip we can read from multiple arrays at the same time.
        row = int(row)
        column = int(column)
        peak_score = float(peak_score)

        # Convert the raster-cell centre to map coordinates, where we care about the centre of each cell.
        x, y = transform * (
            column + 0.5,
            row + 0.5,
        )
        x = float(x)
        y = float(y)

        if _too_close_to_selected(
            x=x,
            y=y,
            selected=selected,
            suppression_radius_m=suppression_radius_m,
        ):
            continue #if it is too close to an already accepted viewpoint, skip.

        candidate = ViewpointRegion(
            identifier=f"generated_{len(selected) + 1}",
            x=x,
            y=y,
            radius_m=verification_radius_m,
        )

        stats = _neighbourhood_statistics(
            opf_result=opf_result,
            viewpoint=candidate,
            peak_score=peak_score,
            support_tolerance=support_tolerance,
        )

        if stats.count < min_neighbourhood_cells:
            continue

        if stats.support_fraction < min_support_fraction:
            continue

        selected.append(
            ViewpointGenerationResult(
                viewpoint=candidate,
                peak_score=peak_score,
                local_mean=stats.mean,
                local_std=stats.std,
                support_fraction=stats.support_fraction,
                neighbourhood_cells=stats.count,
                row=row,
                column=column,
            )
        )

        if len(selected) >= max_viewpoints:
            break

    return selected


def _neighbourhood_statistics(
    *,
    opf_result: ObservabilityPotentialFieldResult,
    viewpoint: ViewpointRegion,
    peak_score: float,
    support_tolerance: float,
) -> _NeighbourhoodStatistics:
    """Measure how strongly the existing viewpoint region supports a peak."""

    local_opf = build_viewpoint_opf(
        opf_result=opf_result,
        viewpoint=viewpoint,
    ) #cutout of OPF within a given radius.

    values = np.asarray(
        local_opf.field[local_opf.valid_mask],
        dtype=np.float32,
    ) #filter out invalid cells, obtain 1D array.

    if values.size == 0:
        return _NeighbourhoodStatistics(
            mean=float("nan"),
            std=float("nan"),
            support_fraction=0.0,
            count=0,
        )

    support_threshold = peak_score - support_tolerance

    return _NeighbourhoodStatistics(
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        support_fraction=float(
            np.count_nonzero(values >= support_threshold)
            / values.size
        ),
        count=int(values.size),
    ) 


def _too_close_to_selected(
    *,
    x: float,
    y: float,
    selected: list[ViewpointGenerationResult],
    suppression_radius_m: float,
) -> bool:
    """Return True when a candidate is too close to an accepted viewpoint."""

    radius_squared = suppression_radius_m**2

    for result in selected:
        viewpoint = result.viewpoint

        distance_squared = (
            (x - viewpoint.x) ** 2
            + (y - viewpoint.y) ** 2
        )

        if distance_squared < radius_squared:
            return True

    return False


def _validate_parameters(
    *,
    opf_result: ObservabilityPotentialFieldResult,
    verification_radius_m: float,
    suppression_radius_m: float,
    max_viewpoints: int,
    support_tolerance: float,
    min_support_fraction: float,
    min_neighbourhood_cells: int,
    minimum_score: float | None,
) -> None:
    if not np.isfinite(verification_radius_m) or verification_radius_m <= 0:
        raise ValueError(
            "verification_radius_m must be a finite value greater than zero."
        )

    if not np.isfinite(suppression_radius_m) or suppression_radius_m < 0:
        raise ValueError(
            "suppression_radius_m must be a finite non-negative value."
        )

    if max_viewpoints < 1:
        raise ValueError(
            "max_viewpoints must be at least 1."
        )

    if not np.isfinite(support_tolerance) or support_tolerance < 0:
        raise ValueError(
            "support_tolerance must be a finite non-negative value."
        )

    if (
        not np.isfinite(min_support_fraction)
        or not 0.0 <= min_support_fraction <= 1.0
    ):
        raise ValueError(
            "min_support_fraction must lie between 0 and 1."
        )

    if min_neighbourhood_cells < 1:
        raise ValueError(
            "min_neighbourhood_cells must be at least 1."
        )

    if minimum_score is not None and not np.isfinite(minimum_score):
        raise ValueError(
            "minimum_score must be finite when supplied."
        )

    crs = opf_result.crs

    if not crs.is_projected:
        raise ValueError(
            "Viewpoint generation requires a projected CRS."
        )

    _, unit_to_metre = crs.linear_units_factor

    # ViewpointRegion.radius_m is currently compared directly with OPF
    # x/y coordinates by build_viewpoint_opf(), so the CRS must use metres.
    if not np.isclose(unit_to_metre, 1.0):
        raise ValueError(
            "Viewpoint generation currently requires OPF coordinates "
            "whose linear unit is metres."
        )