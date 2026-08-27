"""
Generate Rivelero's initial 3D visibility field using GDAL viewsheds.

The result is a scalar surface:

    (x, y) -> z = visibility potential

where:
    x, y = the projected position of a possible observer cell
    z    = the fraction of sampled target-region cells visible from that cell

This module deliberately contains only terrain visibility. It does not yet
include NDVI, distance attenuation, obstacles, image quality, candidate CSVs,
or final multi-module weighting.

Requirements
------------
    numpy
    rasterio
    GDAL Python bindings (``from osgeo import gdal``)

Important
---------
The target and observer geometries must be expressed in the DEM's projected
coordinate reference system.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask

try:
    from osgeo import gdal
except ImportError as error:
    gdal = None
    _GDAL_IMPORT_ERROR = error
else:
    _GDAL_IMPORT_ERROR = None


Geometry = Mapping[str, Any]
ProgressCallback = Callable[[int, int], None]


@dataclass(slots=True)
class VisibilityFieldResult:
    """
    A georeferenced 3D visibility surface.

    ``field`` is a 2D raster whose cell value is the field height z.
    The affine transform supplies each cell's x and y position.

    Values:
        0.0 = none of the sampled target region is visible
        1.0 = all of the sampled target region is visible
        NaN = outside the permitted observer region or invalid DEM
    """

    field: np.ndarray
    transform: Affine
    crs: CRS

    target_mask: np.ndarray
    observer_mask: np.ndarray

    sampled_target_count: int
    observer_height_m: float
    target_height_m: float
    max_distance_m: float
    target_spacing_cells: int

    @property
    def z(self) -> np.ndarray:
        """Return the field heights."""
        return self.field

    def xyz_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return complete X, Y and Z grids suitable for a 3D surface plot.

        The coordinates refer to raster-cell centres. The calculation supports
        affine transforms containing rotation or shear, not merely north-up
        rasters.
        """
        height, width = self.field.shape

        columns, rows = np.meshgrid(
            np.arange(width, dtype=np.float64) + 0.5,
            np.arange(height, dtype=np.float64) + 0.5,
        )

        x = (
            self.transform.a * columns
            + self.transform.b * rows
            + self.transform.c
        )
        y = (
            self.transform.d * columns
            + self.transform.e * rows
            + self.transform.f
        )

        return x, y, self.field

    def as_display_mapping(self) -> dict[str, Any]:
        """
        Return a mapping accepted by the GUI's ``coerce_raster_result()``.
        """
        return {
            "data": self.field,
            "transform": self.transform,
            "crs": self.crs,
            "title": "Visibility Field",
            "colour_map": "YlOrBr",
            "colourbar_label": "Visibility field height",
            "vmin": 0.0,
            "vmax": 1.0,
        }


def build_visibility_field(
    *,
    dem_path: str | Path,
    target_geometry: Geometry | Sequence[Geometry],
    observer_geometry: Geometry | Sequence[Geometry] | None = None,
    observer_height_m: float = 1.75,
    target_height_m: float = 0.0,
    max_distance_m: float = 500.0,
    target_spacing_cells: int = 10,
    curvature_coefficient: float = 0.85714,
    all_touched: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> VisibilityFieldResult:
    """
    Build a target-oriented visibility field using GDAL ViewshedGenerate.

    Parameters
    ----------
    dem_path:
        Projected DEM GeoTIFF.

    target_geometry:
        Polygon or polygons delimiting where the object/species may occur.
        Coordinates must use the DEM's CRS.

    observer_geometry:
        Optional polygon or polygons delimiting where an observer may stand.
        If omitted, every valid DEM cell may be an observer position.

    observer_height_m:
        Height of the real observer above the terrain. This is applied at
        every possible observer cell.

    target_height_m:
        Height of the real target above the terrain.

    max_distance_m:
        Maximum observer-target distance evaluated by GDAL.

    target_spacing_cells:
        Sample one target cell every N rows and N columns. Set to 1 to use
        every target cell. Larger values are faster but less detailed.

    curvature_coefficient:
        GDAL curvature/refraction coefficient. 0.85714 represents the usual
        visible-light approximation for an Earth CRS.

    all_touched:
        Include every raster cell touched by the region polygon.

    progress_callback:
        Optional function called as ``progress_callback(completed, total)``.

    Returns
    -------
    VisibilityFieldResult
        A georeferenced surface. Its field value z at each observer position
        (x, y) is the proportion of sampled target cells visible from there.

    Method
    ------
    For each sampled target cell, this function runs a reverse viewshed.
    Terrain line of sight is tested along the same segment in reverse, so the
    endpoint heights are swapped:

        GDAL temporary observer height = real target height
        GDAL temporary target height   = real observer height

    The binary viewsheds are accumulated. Dividing by the number of sampled
    target cells produces a field bounded between zero and one.
    """
    _require_gdal()
    _validate_parameters(
        observer_height_m=observer_height_m,
        target_height_m=target_height_m,
        max_distance_m=max_distance_m,
        target_spacing_cells=target_spacing_cells,
        curvature_coefficient=curvature_coefficient,
    )

    dem_path = Path(dem_path)

    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM does not exist: {dem_path}")

    with rasterio.open(dem_path) as source:

        if source.count < 1:
            raise ValueError("The DEM contains no raster bands.")

  
        coordinate_mode = (
            source.tags()
            .get("coordinate_mode", "")
            .strip()
            .lower()
        )
        source_crs = source.crs

        if coordinate_mode != "unreal_local":
            if source_crs is None:
                raise ValueError(
                    "The DEM has no coordinate reference system."
                )

            if not source_crs.is_projected:
                raise ValueError(
                    "GDAL viewsheds require a projected DEM for meaningful "
                    "distance and visibility results."
                )

        # Unreal-local coordinates are flat Cartesian metre coordinates, so
        # Earth curvature/refraction must not be applied.
        effective_curvature_coefficient = (
            0.0
            if coordinate_mode == "unreal_local"
            else curvature_coefficient
        )


        dem = source.read(1, masked=True).astype(np.float64)
        dem_shape = (source.height, source.width)
        dem_transform = source.transform
        dem_crs = source.crs

    valid_dem = ~np.ma.getmaskarray(dem)
    valid_dem &= np.isfinite(np.asarray(dem.filled(np.nan)))

    if not valid_dem.any():
        raise ValueError("The DEM contains no valid elevation cells.")

    if not valid_dem.all():
        warnings.warn(
            "The DEM contains NoData cells. GDAL does not currently perform "
            "special terrain processing for source NoData during viewshed "
            "generation, so viewsheds crossing those gaps may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    target_mask = _geometry_to_mask(
        target_geometry,
        shape=dem_shape,
        transform=dem_transform,
        all_touched=all_touched,
    )
    target_mask &= valid_dem

    observer_mask = _geometry_to_mask(
        observer_geometry,
        shape=dem_shape,
        transform=dem_transform,
        all_touched=all_touched,
    )
    observer_mask &= valid_dem

    if not target_mask.any():
        raise ValueError(
            "The target region does not contain any valid DEM cells."
        )

    if not observer_mask.any():
        raise ValueError(
            "The observer region does not contain any valid DEM cells."
        )

    target_cells = _sample_target_cells(
        target_mask,
        spacing=target_spacing_cells,
    )
    total_targets = len(target_cells)

    visible_count = np.zeros(dem_shape, dtype=np.float64)

    gdal.UseExceptions()

    #for Unreal-local DEMs, ViewshedGenerate receives a temporary
    # in-memory copy with the CRS removed. Older GDAL versions try to obtain an
    # ellipsoid from any attached CRS; an Unreal engineering CRS has none.
    #
    # The GeoTransform is retained, so GDAL still has the local X/Y geometry
    # and metre pixel spacing needed by the viewshed algorithm.
    dataset = _open_gdal_viewshed_dataset(
        dem_path,
        strip_crs=(coordinate_mode == "unreal_local"),
    )


    try:
        band = dataset.GetRasterBand(1)

        if band is None:
            raise RuntimeError("GDAL could not access DEM band 1.")

        for completed, (row, column) in enumerate(target_cells, start=1):
            target_x, target_y = _cell_centre(
                dem_transform,
                row=row,
                column=column,
            )

            reverse_viewshed = _run_gdal_viewshed(
                band=band,
                observer_x=target_x,
                observer_y=target_y,
                # Reverse traversal: preserve the physical endpoint heights.
                temporary_observer_height=target_height_m,
                temporary_target_height=observer_height_m,
                max_distance_m=max_distance_m,

                curvature_coefficient=effective_curvature_coefficient,

            )

            try:
                _accumulate_viewshed(
                    accumulator=visible_count,
                    viewshed_dataset=reverse_viewshed,
                    dem_transform=dem_transform,
                )
            finally:
                reverse_viewshed = None

            if progress_callback is not None:
                progress_callback(completed, total_targets)
    finally:
        dataset = None

    field = visible_count / float(total_targets)
    field = np.clip(field, 0.0, 1.0)

    # Z is undefined where the user has forbidden observer positions.
    field = np.where(observer_mask, field, np.nan).astype(np.float32)

    return VisibilityFieldResult(
        field=field,
        transform=dem_transform,
        crs=dem_crs,
        target_mask=target_mask,
        observer_mask=observer_mask,
        sampled_target_count=total_targets,
        observer_height_m=float(observer_height_m),
        target_height_m=float(target_height_m),
        max_distance_m=float(max_distance_m),
        target_spacing_cells=int(target_spacing_cells),
    )


def save_visibility_field(
    result: VisibilityFieldResult,
    output_path: str | Path,
    *,
    nodata: float = -9999.0,
) -> Path:
    """Save the field heights as a georeferenced Float32 GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = np.where(
        np.isfinite(result.field),
        result.field,
        nodata,
    ).astype(np.float32)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output.shape[0],
        width=output.shape[1],
        count=1,
        dtype="float32",
        crs=result.crs,
        transform=result.transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(output, 1)
        destination.set_band_description(1, "visibility_field_height")
        destination.update_tags(
            sampled_target_count=result.sampled_target_count,
            observer_height_m=result.observer_height_m,
            target_height_m=result.target_height_m,
            max_distance_m=result.max_distance_m,
            target_spacing_cells=result.target_spacing_cells,
            field_min=0.0,
            field_max=1.0,
        )

    return output_path




def _open_gdal_viewshed_dataset(
    dem_path: Path,
    *,
    strip_crs: bool,
) -> Any:
    """
    Open a DEM for GDAL ViewshedGenerate.

    Normal GIS DEMs are returned directly. For an Unreal-local DEM, a temporary
    GDAL MEM copy is created with the same raster values and GeoTransform but
    with no CRS. This prevents GDAL/PROJ from trying to obtain an Earth
    ellipsoid from an engineering/local Unreal CRS.
    """
    source_dataset = gdal.Open(str(dem_path), gdal.GA_ReadOnly)

    if source_dataset is None:
        raise RuntimeError(f"GDAL could not open DEM: {dem_path}")

    if not strip_crs:
        return source_dataset

    source_band = source_dataset.GetRasterBand(1)

    if source_band is None:
        source_dataset = None
        raise RuntimeError("GDAL could not access DEM band 1.")

    memory_driver = gdal.GetDriverByName("MEM")

    if memory_driver is None:
        source_dataset = None
        raise RuntimeError("GDAL MEM driver is unavailable.")

    local_dataset = memory_driver.Create(
        "",
        source_dataset.RasterXSize,
        source_dataset.RasterYSize,
        1,
        source_band.DataType,
    )

    if local_dataset is None:
        source_dataset = None
        raise RuntimeError(
            "GDAL could not create the temporary Unreal-local DEM."
        )

    # Preserve the local affine coordinates, but deliberately do NOT copy
    # source_dataset.GetSpatialRef().
    local_dataset.SetGeoTransform(source_dataset.GetGeoTransform())

    local_band = local_dataset.GetRasterBand(1)

    if local_band is None:
        source_dataset = None
        local_dataset = None
        raise RuntimeError(
            "GDAL could not access the temporary Unreal-local DEM band."
        )

    source_array = source_band.ReadAsArray()

    if source_array is None:
        source_dataset = None
        local_dataset = None
        raise RuntimeError("GDAL could not read the Unreal-local DEM.")

    local_band.WriteArray(source_array)

    nodata = source_band.GetNoDataValue()
    if nodata is not None:
        local_band.SetNoDataValue(nodata)

    local_band.FlushCache()
    local_dataset.FlushCache()

    # The in-memory dataset is now independent of the file-backed source.
    source_dataset = None

    return local_dataset


def _run_gdal_viewshed(
    *,
    band: Any,
    observer_x: float,
    observer_y: float,
    temporary_observer_height: float,
    temporary_target_height: float,
    max_distance_m: float,
    curvature_coefficient: float,
) -> Any:
    """
    Run one binary GDAL viewshed and return its in-memory dataset.

    ``ViewshedGenerate`` added an extra-options argument in newer GDAL
    versions, so the function tries the modern call and then the older
    compatible signature.
    """
    mode = gdal.GVM_Edge
    output_type = gdal.GVOT_NORMAL

    common_arguments = (
        band,
        "MEM",
        "",
        [],
        float(observer_x),
        float(observer_y),
        float(temporary_observer_height),
        float(temporary_target_height),
        1.0,   # visible
        0.0,   # invisible
        0.0,   # outside maximum range
        0.0,   # output NoData value
        float(curvature_coefficient),
        mode,
        float(max_distance_m),
        None,  # progress callback
        None,  # callback data
        output_type,
    )

    try:
        viewshed = gdal.ViewshedGenerate(
            *common_arguments,
            [],  # modern GDAL extra options
        )
    except TypeError:
        viewshed = gdal.ViewshedGenerate(*common_arguments)

    if viewshed is None:
        raise RuntimeError(
            "GDAL ViewshedGenerate returned no output dataset."
        )

    return viewshed


def _accumulate_viewshed(
    *,
    accumulator: np.ndarray,
    viewshed_dataset: Any,
    dem_transform: Affine,
) -> None:
    """
    Insert GDAL's possibly cropped viewshed into the full DEM-sized accumulator.
    """
    viewshed_array = viewshed_dataset.GetRasterBand(1).ReadAsArray()

    if viewshed_array is None:
        raise RuntimeError("GDAL produced a viewshed with no raster data.")

    viewshed_array = np.asarray(viewshed_array)
    viewshed_transform = Affine.from_gdal(
        *viewshed_dataset.GetGeoTransform()
    )

    # Convert the output raster's upper-left corner into DEM pixel coordinates.
    column_float, row_float = (~dem_transform) * (
        viewshed_transform.c,
        viewshed_transform.f,
    )

    destination_column0 = int(round(column_float))
    destination_row0 = int(round(row_float))

    source_height, source_width = viewshed_array.shape

    destination_row1 = destination_row0 + source_height
    destination_column1 = destination_column0 + source_width

    clipped_row0 = max(0, destination_row0)
    clipped_column0 = max(0, destination_column0)
    clipped_row1 = min(accumulator.shape[0], destination_row1)
    clipped_column1 = min(accumulator.shape[1], destination_column1)

    if clipped_row0 >= clipped_row1 or clipped_column0 >= clipped_column1:
        raise RuntimeError(
            "The GDAL viewshed does not overlap the source DEM grid."
        )

    source_row0 = clipped_row0 - destination_row0
    source_column0 = clipped_column0 - destination_column0
    source_row1 = source_row0 + (clipped_row1 - clipped_row0)
    source_column1 = source_column0 + (
        clipped_column1 - clipped_column0
    )

    visible = (
        viewshed_array[
            source_row0:source_row1,
            source_column0:source_column1,
        ]
        == 1
    )

    accumulator[
        clipped_row0:clipped_row1,
        clipped_column0:clipped_column1,
    ] += visible


def _geometry_to_mask(
    geometry: Geometry | Sequence[Geometry] | None,
    *,
    shape: tuple[int, int],
    transform: Affine,
    all_touched: bool,
) -> np.ndarray:
    """Rasterise one or more polygons into a Boolean inclusion mask."""
    if geometry is None:
        return np.ones(shape, dtype=bool)

    geometries = _normalise_geometries(geometry)

    if not geometries:
        raise ValueError("The supplied region contains no geometries.")

    return geometry_mask(
        geometries,
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=all_touched,
    )


def _normalise_geometries(
    geometry: Geometry | Sequence[Geometry],
) -> list[Geometry]:
    """
    Accept a geometry, Feature, FeatureCollection, or sequence of geometries.
    """
    if isinstance(geometry, Mapping):
        geometry_type = geometry.get("type")

        if geometry_type == "Feature":
            nested_geometry = geometry.get("geometry")
            if not isinstance(nested_geometry, Mapping):
                raise ValueError("GeoJSON Feature has no valid geometry.")
            return [nested_geometry]

        if geometry_type == "FeatureCollection":
            features = geometry.get("features", [])
            result: list[Geometry] = []

            for feature in features:
                if not isinstance(feature, Mapping):
                    continue
                nested_geometry = feature.get("geometry")
                if isinstance(nested_geometry, Mapping):
                    result.append(nested_geometry)

            return result

        return [geometry]

    return [
        item
        for item in geometry
        if isinstance(item, Mapping)
    ]


def _sample_target_cells(
    target_mask: np.ndarray,
    *,
    spacing: int,
) -> list[tuple[int, int]]:
    """Sample the target region on a regular raster grid."""
    rows, columns = np.indices(target_mask.shape)

    sampling_grid = (
        (rows % spacing == 0)
        & (columns % spacing == 0)
    )

    sampled_rows, sampled_columns = np.where(
        target_mask & sampling_grid
    )

    if sampled_rows.size == 0:
        # A very small target polygon can fall between sampling-grid nodes.
        first_row, first_column = np.argwhere(target_mask)[0]
        return [(int(first_row), int(first_column))]

    return [
        (int(row), int(column))
        for row, column in zip(
            sampled_rows,
            sampled_columns,
            strict=True,
        )
    ]


def _cell_centre(
    transform: Affine,
    *,
    row: int,
    column: int,
) -> tuple[float, float]:
    """Return the projected centre coordinate of one DEM cell."""
    x, y = transform * (column + 0.5, row + 0.5)
    return float(x), float(y)


def _validate_parameters(
    *,
    observer_height_m: float,
    target_height_m: float,
    max_distance_m: float,
    target_spacing_cells: int,
    curvature_coefficient: float,
) -> None:
    if observer_height_m < 0:
        raise ValueError("Observer height cannot be negative.")

    if target_height_m < 0:
        raise ValueError("Target height cannot be negative.")

    if max_distance_m <= 0:
        raise ValueError("Maximum distance must be greater than zero.")

    if target_spacing_cells < 1:
        raise ValueError(
            "Target spacing must be an integer of at least one cell."
        )

    if not np.isfinite(curvature_coefficient):
        raise ValueError("Curvature coefficient must be finite.")


def _require_gdal() -> None:
    if gdal is None:
        raise ImportError(
            "The GDAL Python bindings are required. Install GDAL so that "
            "'from osgeo import gdal' works in the same Python environment."
        ) from _GDAL_IMPORT_ERROR