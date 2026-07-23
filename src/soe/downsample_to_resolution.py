"""Merge candidate viewpoints into fixed-resolution square regions."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

import numpy as np


@dataclass(frozen=True, slots=True)
class MergedViewpoint:
    """
     One occupied grid square acting as a single enlarged viewpoint.

     The bounds define the actual spatial region. The x and y properties
     return the square centre for code that requires one representative
     coordinate.
    """

    identifier: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    source_indices: tuple[int, ...]

    @property
    def x(self) -> float:
        """Return the centre x-coordinate of the square."""
        return (self.xmin + self.xmax) / 2.0

    @property
    def y(self) -> float:
        """Return the centre y-coordinate of the square."""
        return (self.ymin + self.ymax) / 2.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return the square bounds as xmin, ymin, xmax, ymax."""
        return self.xmin, self.ymin, self.xmax, self.ymax

    @property
    def size(self) -> float:
        """Return the side length of the square."""
        return self.xmax - self.xmin

    @property
    def n_source_viewpoints(self) -> int:
        """Return the number of viewpoints merged into the square."""
        return len(self.source_indices)

def _validate_points(points: np.ndarray) -> np.ndarray:
    """Validate and normalise an array of viewpoint coordinates."""
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "points must be an Nx2 array containing x and y coordinates."
        )

    if len(points) == 0:
        return points

    if not np.all(np.isfinite(points)):
        raise ValueError("points contains NaN or infinite coordinates.")

    return points


def _cell_index(
    coordinate: float,
    *,
    origin: float,
    resolution: float,
) -> int:
    """Return the fixed-grid index containing one coordinate."""
    return floor((coordinate - origin) / resolution)


def downsample_to_resolution(
    points: np.ndarray,
    resolution: float,
    *,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[MergedViewpoint]:
    """
    Merge candidate viewpoints into fixed-resolution square regions.

    Every occupied square becomes one MergedViewpoint. Empty squares are
    omitted.

    Args:
        points:
            Nx2 array of viewpoint coordinates in the form ``[x, y]``.

        resolution:
            Side length of each output square, expressed in the same units
            as the coordinates. For a projected CRS this will normally be
            metres.

        origin_x:
            X-coordinate against which the grid is aligned.

        origin_y:
            Y-coordinate against which the grid is aligned.

    Returns:
        One MergedViewpoint for every occupied square, sorted consistently
        by grid row and column.
    """
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution must be a finite value greater than 0.")

    if not np.isfinite(origin_x) or not np.isfinite(origin_y):
        raise ValueError("Grid origins must be finite coordinates.")

    points = _validate_points(points)

    if len(points) == 0:
        return []

    members_by_cell: dict[tuple[int, int], list[int]] = {}

    for point_index, (x, y) in enumerate(points):
        column = _cell_index(
            float(x),
            origin=origin_x,
            resolution=resolution,
        )

        row = _cell_index(
            float(y),
            origin=origin_y,
            resolution=resolution,
        )

        cell_key = row, column
        members_by_cell.setdefault(cell_key, []).append(point_index)

    merged_viewpoints: list[MergedViewpoint] = []

    for merged_index, ((row, column), source_indices) in enumerate(
        sorted(members_by_cell.items())
    ):
        xmin = origin_x + column * resolution
        ymin = origin_y + row * resolution
        xmax = xmin + resolution
        ymax = ymin + resolution

        merged_viewpoints.append(
            MergedViewpoint(
                identifier=f"merged-viewpoint-{merged_index}",
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                source_indices=tuple(source_indices),
            )
        )

    return merged_viewpoints