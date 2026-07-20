"""Combine Rivelero component fields into an observability potential field."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from .botanical_suitability_field import BotanicalSuitabilityFieldResult
from .obstacle_occlusion_field import ObstacleOcclusionFieldResult
from .visibility_field import VisibilityFieldResult


@dataclass(slots=True)
class ObservabilityPotentialFieldResult:
    """Full-resolution combined observability potential field."""

    field: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    crs: CRS

    visibility_weight: float
    ndvi_weight: float
    occlusion_weight: float

    def as_display_mapping(self) -> dict[str, Any]:
        return {
            "data": self.field,
            "transform": self.transform,
            "crs": self.crs,
            "title": "Observability potential field",
            "colour_map": "viridis",
            "colourbar_label": "Observability potential",
            "vmin": -0.5,
            "vmax": 0.8,
        }


def build_observability_potential_field(
    *,
    visibility_result: VisibilityFieldResult | None = None,
    botanical_result: BotanicalSuitabilityFieldResult | None = None,
    occlusion_result: ObstacleOcclusionFieldResult | None = None,
    visibility_weight: float = 0.5,
    ndvi_weight: float = 0.3,
    occlusion_weight: float = 0.2,
) -> ObservabilityPotentialFieldResult:
    """
    Calculate:

        F = 0.5V + 0.3NDVI - 0.2Occlusion

    The supplied result objects and their arrays are not modified.
    """
    selected_results = [
        result
        for result in (
            visibility_result,
            botanical_result,
            occlusion_result,
        )
        if result is not None
    ]

    if not selected_results:
        raise ValueError(
            "At least one component field is required to build the OPF."
        )

    _validate_matching_grids(*selected_results)

    reference = selected_results[0]
    reference_shape = np.asarray(reference.field).shape

    component_arrays: list[np.ndarray] = []

    if visibility_result is not None:
        visibility = np.asarray(
            visibility_result.field,
            dtype=np.float32,
        )
        component_arrays.append(visibility)

    if botanical_result is not None:
        botanical = np.asarray(
            botanical_result.field,
            dtype=np.float32,
        )
        component_arrays.append(botanical)

    if occlusion_result is not None:
        occlusion = np.asarray(
            occlusion_result.field,
            dtype=np.float32,
        )
        component_arrays.append(occlusion)

    # A cell is valid only when every selected component is valid there.
    valid_mask = np.ones(reference_shape, dtype=bool)

    for component in component_arrays:
        valid_mask &= np.isfinite(component)

    field = np.full(
        reference_shape,
        np.nan,
        dtype=np.float32,
    )

    field[valid_mask] = 0.0

    if visibility_result is not None:
        field[valid_mask] += (
            visibility_weight * visibility[valid_mask]
        )

    if botanical_result is not None:
        field[valid_mask] += (
            ndvi_weight * botanical[valid_mask]
        )

    if occlusion_result is not None:
        field[valid_mask] -= (
            occlusion_weight * occlusion[valid_mask]
        )

    return ObservabilityPotentialFieldResult(
        field=field,
        valid_mask=valid_mask,
        transform=reference.transform,
        crs=reference.crs,
        visibility_weight=(
            float(visibility_weight)
            if visibility_result is not None
            else 0.0
        ),
        ndvi_weight=(
            float(ndvi_weight)
            if botanical_result is not None
            else 0.0
        ),
        occlusion_weight=(
            float(occlusion_weight)
            if occlusion_result is not None
            else 0.0
        ),
    )

def _validate_matching_grids(
    *results: object,
) -> None:
    """Require selected fields to have identical grids."""

    if not results:
        raise ValueError("No component results were supplied.")

    reference = results[0]
    reference_field = np.asarray(reference.field)

    for result in results[1:]:
        if np.asarray(result.field).shape != reference_field.shape:
            raise ValueError(
                "The selected component fields have different dimensions. "
                "Run them on the same DEM crop."
            )

        if result.transform != reference.transform:
            raise ValueError(
                "The selected component fields have different affine "
                "transforms. Run them on the same DEM crop."
            )

        if result.crs != reference.crs:
            raise ValueError(
                "The selected component fields have different "
                "coordinate systems."
            )