"""Viewpoint models and viewpoint-specific OPF extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from affine import Affine
import numpy as np
from rasterio.crs import CRS
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .observability_potential_field import (
    ObservabilityPotentialFieldResult,
)



@dataclass(frozen=True, slots=True)
class ViewpointRegion:
    """One circular or rectangular local OPF region."""

    identifier: str
    x: float
    y: float
    radius_m: float | None = None
    bounds: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class ViewpointOPFResult:
    """The portion of an OPF available to one viewpoint."""

    
    viewpoint: ViewpointRegion
    field: np.ndarray
    valid_mask: np.ndarray
    transform: Affine
    crs: CRS

    def as_display_mapping(self) -> dict[str, Any]:
        return {
            "data": self.field,
            "transform": self.transform,
            "crs": self.crs,
            "title": f"OPF around {self.viewpoint.identifier}",
            "colour_map": "viridis",
            "colourbar_label": "Observability potential",
            "vmin": -0.5,
            "vmax": 0.8,
        }


def build_viewpoint_opf(
    *,
    opf_result: ObservabilityPotentialFieldResult,

    viewpoint: ViewpointRegion,
) -> ViewpointOPFResult:
 
    """Mask an existing OPF to one circular or rectangular region."""

    has_radius = viewpoint.radius_m is not None
    has_bounds = viewpoint.bounds is not None

    if has_radius == has_bounds:
        raise ValueError(
            "A viewpoint region must define exactly one of radius_m or bounds."
        )

    rows, columns = np.indices(
        opf_result.field.shape,
        dtype=np.float64,
    )

    # Convert pixel centres to coordinates in the OPF/DEM CRS.
    pixel_columns = columns + 0.5
    pixel_rows = rows + 0.5
    transform = opf_result.transform

    x_coordinates = (
        transform.a * pixel_columns
        + transform.b * pixel_rows
        + transform.c
    )

    y_coordinates = (
        transform.d * pixel_columns
        + transform.e * pixel_rows
        + transform.f
    )


    if viewpoint.bounds is not None:
        xmin, ymin, xmax, ymax = viewpoint.bounds

        if not np.isfinite((xmin, ymin, xmax, ymax)).all():
            raise ValueError(
                "Viewpoint bounds must contain only finite values."
            )

        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "Viewpoint bounds must satisfy xmin < xmax and ymin < ymax."
            )

        region_mask = (
            (x_coordinates >= xmin)
            & (x_coordinates < xmax)
            & (y_coordinates >= ymin)
            & (y_coordinates < ymax)
        )
    else:
        radius_m = viewpoint.radius_m

        if radius_m is None:
            raise ValueError("Viewpoint radius is missing.")

        if not np.isfinite(radius_m) or radius_m <= 0:
            raise ValueError(
                "Viewpoint radius must be a finite value greater than zero."
            )

        region_mask = (
            (x_coordinates - viewpoint.x) ** 2
            + (y_coordinates - viewpoint.y) ** 2
            <= radius_m**2
        )

    # Anything already invalid in the OPF stays invalid.
    valid_mask = opf_result.valid_mask & region_mask

    valid_rows, valid_columns = np.nonzero(valid_mask)

    if valid_rows.size == 0:
       
        raise ValueError(
            "The viewpoint region contains no valid OPF cells."
        )

    row_min = int(valid_rows.min())
    row_max = int(valid_rows.max())
    column_min = int(valid_columns.min())
    column_max = int(valid_columns.max())

    row_slice = slice(row_min, row_max + 1)
    column_slice = slice(column_min, column_max + 1)

    cropped_mask = valid_mask[row_slice, column_slice]
    cropped_field = opf_result.field[
        row_slice,
        column_slice,
    ].astype(
        np.float32,
        copy=True,
    )

    cropped_field[~cropped_mask] = np.nan

    window = Window(
        col_off=column_min,
        row_off=row_min,
        width=column_max - column_min + 1,
        height=row_max - row_min + 1,
    )

    cropped_transform = window_transform(
        window,
        opf_result.transform,
    )

    return ViewpointOPFResult(
        viewpoint=viewpoint,
        field=cropped_field,
        valid_mask=cropped_mask,
        transform=cropped_transform,
        crs=opf_result.crs,
    )