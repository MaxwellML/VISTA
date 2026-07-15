"""
Generate Rivelero's obstacle-occlusion field from OpenStreetMap features.

The result is a georeferenced scalar surface:

    (x, y) -> z = local obstacle-coverage fraction

where:
    x, y = projected position of a possible observer cell
    z    = the fraction of a local square neighbourhood occupied by mapped
           obstacle geometries, bounded between 0 and 1

The output is aligned exactly to the supplied DEM so it can be combined
cell-by-cell with ``visibility_field.py`` and
``botanical_suitability_field.py``.

Scientific interpretation
-------------------------
This is an obstacle-density / occlusion-risk proxy. It is not a true 3D
line-of-sight occlusion calculation because OpenStreetMap geometries often lack
complete and reliable height information. Terrain occlusion remains the job of
the GDAL visibility field.

Requirements
------------
    numpy
    rasterio
    geopandas
    shapely
    affine
    osmnx          (required only when downloading features)

The ``features_gdf`` argument allows pre-fetched or synthetic features to be
used without OSMnx or a network connection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

try:
    import osmnx as ox
except ImportError as error:
    ox = None
    _OSMNX_IMPORT_ERROR = error
else:
    _OSMNX_IMPORT_ERROR = None


Geometry = Mapping[str, Any]


DEFAULT_OBSTACLE_TAGS: dict[str, Any] = {
    "building": True,
    "barrier": True,
    "bridge": True,
    "tunnel": True,
    "man_made": [
        "tower",
        "mast",
        "chimney",
        "silo",
        "storage_tank",
        "water_tower",
        "crane",
        "antenna",
        "bridge",
        "tunnel",
    ],
    "power": ["tower", "pole"],
}


__all__ = [
    "DEFAULT_OBSTACLE_TAGS",
    "ObstacleOcclusionFieldResult",
    "build_obstacle_occlusion_field",
    "fetch_obstacles_for_extent",
    "projected_bounds_to_wgs84",
    "save_obstacle_occlusion_field",
]


@dataclass(slots=True)
class ObstacleOcclusionFieldResult:
    """
    A georeferenced 3D obstacle-occlusion surface.

    ``field`` is the local obstacle-coverage fraction and therefore the field
    height z.

    Values:
        0.0 = no mapped obstacle coverage in the local neighbourhood
        1.0 = the local neighbourhood is completely obstacle-covered
        NaN = outside the permitted field region or invalid DEM

    ``obstacle_mask`` is the unsmoothed binary raster:
        False / 0 = no obstacle footprint
        True  / 1 = obstacle footprint or buffered point/line feature
    """

    field: np.ndarray
    obstacle_mask: np.ndarray
    field_mask: np.ndarray

    transform: Affine
    crs: CRS

    raw_features: gpd.GeoDataFrame
    prepared_features: gpd.GeoDataFrame
    query_bbox_wgs84: tuple[float, float, float, float]

    neighbourhood_radius_m: float
    point_buffer_m: float
    line_buffer_m: float
    use_bounding_boxes: bool
    feature_count: int

    @property
    def z(self) -> np.ndarray:
        """Return the obstacle-occlusion field heights."""
        return self.field

    @property
    def occlusion_penalty(self) -> np.ndarray:
        """Descriptive alias for the field."""
        return self.field

    def xyz_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return projected X, Y and Z grids for a 3D surface plot.

        X and Y are cell-centre coordinates. Affine rotation and shear are
        supported.
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
            "title": "Obstacle occlusion field",
            "colour_map": "magma",
            "colourbar_label": "Local obstacle coverage",
            "vmin": 0.0,
            "vmax": 1.0,
        }


def build_obstacle_occlusion_field(
    *,
    dem_path: str | Path,
    field_geometry: Geometry | Sequence[Geometry] | None = None,
    tags: dict[str, Any] | None = None,
    neighbourhood_radius_m: float = 50.0,
    point_buffer_m: float = 2.0,
    line_buffer_m: float = 1.0,
    polygon_buffer_m: float = 0.0,
    use_bounding_boxes: bool = False,
    all_touched: bool = True,
    requests_timeout: int = 60,
    use_cache: bool = True,
    features_gdf: gpd.GeoDataFrame | None = None,
) -> ObstacleOcclusionFieldResult:
    """
    Build a DEM-aligned obstacle-occlusion field.

    Parameters
    ----------
    dem_path:
        Projected DEM GeoTIFF defining the output shape, transform and CRS.

    field_geometry:
        Optional polygon or polygons delimiting where field values should be
        produced. Coordinates must use the DEM CRS. If omitted, use every valid
        DEM cell.

    tags:
        OSM tag query. OSMnx treats different tags as a union: a returned
        feature need only match one requested tag.

    neighbourhood_radius_m:
        Half-width of the local square used to calculate obstacle coverage.
        Set to zero to return the direct binary obstacle-occupancy field.

    point_buffer_m:
        Radius used to turn point obstacles such as poles into polygons.

    line_buffer_m:
        Buffer distance used to turn line features such as barriers into
        polygons.

    polygon_buffer_m:
        Optional outward buffer applied to polygon obstacles.

    use_bounding_boxes:
        If True, replace every obstacle geometry with its rectangular envelope,
        reproducing the original module's behaviour. False is recommended
        because envelopes generally overestimate occupied area.

    all_touched:
        Burn every DEM cell touched by an obstacle geometry.

    requests_timeout, use_cache:
        OSMnx/Overpass request settings.

    features_gdf:
        Optional pre-fetched GeoDataFrame. When provided, skip the OSMnx query.
        Its CRS must be defined. This is useful for testing or cached workflows.

    Returns
    -------
    ObstacleOcclusionFieldResult
        A georeferenced field bounded between zero and one.
    """
    _validate_parameters(
        neighbourhood_radius_m=neighbourhood_radius_m,
        point_buffer_m=point_buffer_m,
        line_buffer_m=line_buffer_m,
        polygon_buffer_m=polygon_buffer_m,
        requests_timeout=requests_timeout,
    )

    dem_path = Path(dem_path)

    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM does not exist: {dem_path}")

    with rasterio.open(dem_path) as source:
        if source.count < 1:
            raise ValueError("The DEM contains no raster bands.")

        if source.crs is None:
            raise ValueError("The DEM has no coordinate reference system.")

        if not source.crs.is_projected:
            raise ValueError(
                "Use a projected DEM so distances, buffers and field X/Y "
                "coordinates are expressed in linear units."
            )

        dem = source.read(1, masked=True).astype(np.float64)
        dem_shape = (source.height, source.width)
        dem_transform = source.transform
        dem_crs = source.crs

    valid_dem = ~np.ma.getmaskarray(dem)
    valid_dem &= np.isfinite(np.asarray(dem.filled(np.nan)))

    if not valid_dem.any():
        raise ValueError("The DEM contains no valid cells.")

    field_mask = _geometry_to_mask(
        field_geometry,
        shape=dem_shape,
        transform=dem_transform,
        all_touched=False,
    )
    field_mask &= valid_dem

    if not field_mask.any():
        raise ValueError(
            "The field region does not contain any valid DEM cells."
        )

    query_bounds_projected = _mask_bounds(
        field_mask,
        transform=dem_transform,
    )

    query_bbox_wgs84 = projected_bounds_to_wgs84(
        left=query_bounds_projected[0],
        bottom=query_bounds_projected[1],
        right=query_bounds_projected[2],
        top=query_bounds_projected[3],
        projected_crs=dem_crs,
    )

    if features_gdf is None:
        raw_features = fetch_obstacles_for_extent(
            left=query_bounds_projected[0],
            bottom=query_bounds_projected[1],
            right=query_bounds_projected[2],
            top=query_bounds_projected[3],
            projected_crs=dem_crs,
            tags=tags,
            requests_timeout=requests_timeout,
            use_cache=use_cache,
        )
    else:
        raw_features = _normalise_supplied_features(
            features_gdf,
            destination_crs=dem_crs,
        )

    raw_features = _clip_features_to_bounds(
        raw_features,
        query_bounds_projected,
    )

    prepared_features = _prepare_obstacle_features(
        raw_features,
        point_buffer_m=point_buffer_m,
        line_buffer_m=line_buffer_m,
        polygon_buffer_m=polygon_buffer_m,
        use_bounding_boxes=use_bounding_boxes,
    )

    obstacle_mask = _rasterize_obstacles(
        prepared_features,
        shape=dem_shape,
        transform=dem_transform,
        all_touched=all_touched,
    )

    obstacle_mask &= field_mask

    field = _local_coverage_fraction(
        obstacle_mask=obstacle_mask,
        valid_mask=field_mask,
        transform=dem_transform,
        radius_m=neighbourhood_radius_m,
    )

    field = np.where(field_mask, field, np.nan).astype(np.float32)

    return ObstacleOcclusionFieldResult(
        field=field,
        obstacle_mask=obstacle_mask,
        field_mask=field_mask,
        transform=dem_transform,
        crs=dem_crs,
        raw_features=raw_features,
        prepared_features=prepared_features,
        query_bbox_wgs84=query_bbox_wgs84,
        neighbourhood_radius_m=float(neighbourhood_radius_m),
        point_buffer_m=float(point_buffer_m),
        line_buffer_m=float(line_buffer_m),
        use_bounding_boxes=bool(use_bounding_boxes),
        feature_count=len(prepared_features),
    )


def projected_bounds_to_wgs84(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    projected_crs: CRS | str,
) -> tuple[float, float, float, float]:
    """
    Convert projected bounds into (left, bottom, right, top) in EPSG:4326.
    """
    return tuple(
        float(value)
        for value in transform_bounds(
            projected_crs,
            "EPSG:4326",
            left,
            bottom,
            right,
            top,
            densify_pts=21,
        )
    )


def fetch_obstacles_for_extent(
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
    projected_crs: CRS | str,
    tags: dict[str, Any] | None = None,
    requests_timeout: int = 60,
    use_cache: bool = True,
) -> gpd.GeoDataFrame:
    """
    Query OSM obstacle-like features and reproject them to ``projected_crs``.

    An empty GeoDataFrame is returned when Overpass reports that no matching
    features exist. Network, timeout and other unexpected errors are raised.
    """
    if ox is None:
        raise ImportError(
            "OSMnx is required to download obstacle features. Install osmnx "
            "or pass a pre-fetched GeoDataFrame through features_gdf."
        ) from _OSMNX_IMPORT_ERROR

    tags = tags or DEFAULT_OBSTACLE_TAGS

    ox.settings.requests_timeout = int(requests_timeout)
    ox.settings.use_cache = bool(use_cache)
    ox.settings.log_console = False

    bbox_wgs84 = projected_bounds_to_wgs84(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        projected_crs=projected_crs,
    )

    try:
        raw = ox.features.features_from_bbox(
            bbox=bbox_wgs84,
            tags=tags,
        )
    except Exception as error:
        # OSMnx versions use different exception classes for an empty result.
        if error.__class__.__name__ in {
            "InsufficientResponseError",
            "EmptyOverpassResponse",
        }:
            return _empty_geodataframe(projected_crs)
        raise

    if raw.empty:
        return _empty_geodataframe(projected_crs)

    raw = raw.reset_index(drop=False)

    if raw.crs is None:
        raw = raw.set_crs("EPSG:4326")

    return raw.to_crs(projected_crs)


def save_obstacle_occlusion_field(
    result: ObstacleOcclusionFieldResult,
    output_path: str | Path,
    *,
    nodata: float = -9999.0,
    include_binary_mask: bool = True,
) -> Path:
    """
    Save the occlusion field as a GeoTIFF.

    Band 1 is the 0..1 field. If ``include_binary_mask`` is True, band 2 stores
    the direct obstacle footprint mask.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_output = np.where(
        np.isfinite(result.field),
        result.field,
        nodata,
    ).astype(np.float32)

    band_count = 2 if include_binary_mask else 1

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=field_output.shape[0],
        width=field_output.shape[1],
        count=band_count,
        dtype="float32",
        crs=result.crs,
        transform=result.transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(field_output, 1)
        destination.set_band_description(
            1,
            "obstacle_occlusion_field_height",
        )

        if include_binary_mask:
            mask_output = np.where(
                result.field_mask,
                result.obstacle_mask.astype(np.float32),
                nodata,
            )
            destination.write(mask_output.astype(np.float32), 2)
            destination.set_band_description(
                2,
                "binary_obstacle_footprint",
            )

        destination.update_tags(
            source="OpenStreetMap via OSMnx",
            neighbourhood_radius_m=result.neighbourhood_radius_m,
            point_buffer_m=result.point_buffer_m,
            line_buffer_m=result.line_buffer_m,
            use_bounding_boxes=result.use_bounding_boxes,
            feature_count=result.feature_count,
            field_min=0.0,
            field_max=1.0,
        )

    return output_path


def _prepare_obstacle_features(
    features: gpd.GeoDataFrame,
    *,
    point_buffer_m: float,
    line_buffer_m: float,
    polygon_buffer_m: float,
    use_bounding_boxes: bool,
) -> gpd.GeoDataFrame:
    """Convert heterogeneous OSM geometries into rasterizable obstacle areas."""
    if features.empty:
        return _empty_geodataframe(features.crs)

    prepared_rows: list[dict[str, Any]] = []

    for _, row in features.iterrows():
        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        prepared = _prepare_geometry(
            geometry,
            point_buffer_m=point_buffer_m,
            line_buffer_m=line_buffer_m,
            polygon_buffer_m=polygon_buffer_m,
            use_bounding_boxes=use_bounding_boxes,
        )

        if prepared is None or prepared.is_empty:
            continue

        values = row.to_dict()
        values["geometry"] = prepared
        prepared_rows.append(values)

    if not prepared_rows:
        return _empty_geodataframe(features.crs)

    return gpd.GeoDataFrame(
        prepared_rows,
        geometry="geometry",
        crs=features.crs,
    )


def _prepare_geometry(
    geometry: BaseGeometry,
    *,
    point_buffer_m: float,
    line_buffer_m: float,
    polygon_buffer_m: float,
    use_bounding_boxes: bool,
) -> BaseGeometry | None:
    """Turn a point, line or polygon into an obstacle-area geometry."""
    if use_bounding_boxes:
        geometry = geometry.envelope

    geometry_type = geometry.geom_type

    if geometry_type in {"Point", "MultiPoint"}:
        prepared = geometry.buffer(point_buffer_m)
    elif geometry_type in {"LineString", "MultiLineString"}:
        prepared = geometry.buffer(line_buffer_m)
    elif geometry_type in {"Polygon", "MultiPolygon"}:
        prepared = (
            geometry.buffer(polygon_buffer_m)
            if polygon_buffer_m > 0
            else geometry
        )
    elif geometry_type == "GeometryCollection":
        buffer_distance = max(
            point_buffer_m,
            line_buffer_m,
            polygon_buffer_m,
        )
        prepared = (
            geometry.buffer(buffer_distance)
            if buffer_distance > 0
            else geometry
        )
    else:
        return None

    if not prepared.is_valid:
        prepared = prepared.buffer(0)

    return prepared


def _rasterize_obstacles(
    features: gpd.GeoDataFrame,
    *,
    shape: tuple[int, int],
    transform: Affine,
    all_touched: bool,
) -> np.ndarray:
    """Burn obstacle areas onto the DEM grid."""
    if features.empty:
        return np.zeros(shape, dtype=bool)

    geometries = [
        geometry
        for geometry in features.geometry
        if geometry is not None and not geometry.is_empty
    ]

    if not geometries:
        return np.zeros(shape, dtype=bool)

    raster = rasterize(
        geometries,
        out_shape=shape,
        fill=0,
        default_value=1,
        transform=transform,
        all_touched=all_touched,
        dtype="uint8",
    )

    return raster.astype(bool)


def _local_coverage_fraction(
    *,
    obstacle_mask: np.ndarray,
    valid_mask: np.ndarray,
    transform: Affine,
    radius_m: float,
) -> np.ndarray:
    """
    Calculate the obstacle-covered fraction of a square around every cell.

    Integral images make this O(H*W), rather than scanning a neighbourhood
    separately for every raster cell.
    """
    if radius_m == 0:
        return obstacle_mask.astype(np.float64)

    pixel_width = float(np.hypot(transform.a, transform.d))
    pixel_height = float(np.hypot(transform.b, transform.e))

    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError("The DEM transform has an invalid pixel size.")

    radius_columns = max(1, int(np.ceil(radius_m / pixel_width)))
    radius_rows = max(1, int(np.ceil(radius_m / pixel_height)))

    obstacle_count = _box_sum(
        obstacle_mask.astype(np.uint8),
        radius_rows=radius_rows,
        radius_columns=radius_columns,
    )
    valid_count = _box_sum(
        valid_mask.astype(np.uint8),
        radius_rows=radius_rows,
        radius_columns=radius_columns,
    )

    fraction = np.zeros(obstacle_mask.shape, dtype=np.float64)

    np.divide(
        obstacle_count,
        valid_count,
        out=fraction,
        where=valid_count > 0,
    )

    return np.clip(fraction, 0.0, 1.0)


def _box_sum(
    array: np.ndarray,
    *,
    radius_rows: int,
    radius_columns: int,
) -> np.ndarray:
    """Return the sum inside a clipped rectangular window around each cell."""
    height, width = array.shape

    integral = np.pad(
        array.astype(np.int64),
        ((1, 0), (1, 0)),
        mode="constant",
    )
    integral = integral.cumsum(axis=0).cumsum(axis=1)

    rows = np.arange(height)
    columns = np.arange(width)

    row0 = np.maximum(0, rows - radius_rows)
    row1 = np.minimum(height, rows + radius_rows + 1)
    column0 = np.maximum(0, columns - radius_columns)
    column1 = np.minimum(width, columns + radius_columns + 1)

    return (
        integral[row1[:, None], column1[None, :]]
        - integral[row0[:, None], column1[None, :]]
        - integral[row1[:, None], column0[None, :]]
        + integral[row0[:, None], column0[None, :]]
    )


def _geometry_to_mask(
    geometry: Geometry | Sequence[Geometry] | None,
    *,
    shape: tuple[int, int],
    transform: Affine,
    all_touched: bool,
) -> np.ndarray:
    """Rasterize the optional field region into a Boolean mask."""
    if geometry is None:
        return np.ones(shape, dtype=bool)

    geometries = _normalise_geometries(geometry)

    if not geometries:
        raise ValueError("The supplied field region has no geometries.")

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
    """Accept a geometry, Feature, FeatureCollection or sequence."""
    if isinstance(geometry, Mapping):
        geometry_type = geometry.get("type")

        if geometry_type == "Feature":
            nested = geometry.get("geometry")

            if not isinstance(nested, Mapping):
                raise ValueError("GeoJSON Feature has no valid geometry.")

            return [nested]

        if geometry_type == "FeatureCollection":
            result: list[Geometry] = []

            for feature in geometry.get("features", []):
                if not isinstance(feature, Mapping):
                    continue

                nested = feature.get("geometry")

                if isinstance(nested, Mapping):
                    result.append(nested)

            return result

        return [geometry]

    return [
        item
        for item in geometry
        if isinstance(item, Mapping)
    ]


def _normalise_supplied_features(
    features: gpd.GeoDataFrame,
    *,
    destination_crs: CRS,
) -> gpd.GeoDataFrame:
    """Validate and reproject injected feature data."""
    if features.crs is None:
        raise ValueError(
            "The supplied features GeoDataFrame must have a CRS."
        )

    if features.empty:
        return _empty_geodataframe(destination_crs)

    return features.to_crs(destination_crs).copy()


def _clip_features_to_bounds(
    features: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
) -> gpd.GeoDataFrame:
    """Intersect geometries with the analysed projected bounding box."""
    if features.empty:
        return features.copy()

    clipping_box = box(*bounds)
    clipped = features.copy()
    clipped["geometry"] = clipped.geometry.intersection(clipping_box)
    clipped = clipped[
        clipped.geometry.notna()
        & ~clipped.geometry.is_empty
    ].copy()

    if clipped.empty:
        return _empty_geodataframe(features.crs)

    return clipped


def _mask_bounds(
    mask: np.ndarray,
    *,
    transform: Affine,
) -> tuple[float, float, float, float]:
    """Return projected bounds around all True cells."""
    rows, columns = np.where(mask)

    if rows.size == 0:
        raise ValueError("Cannot calculate bounds of an empty mask.")

    row0 = int(rows.min())
    row1 = int(rows.max()) + 1
    column0 = int(columns.min())
    column1 = int(columns.max()) + 1

    window = Window(
        col_off=column0,
        row_off=row0,
        width=column1 - column0,
        height=row1 - row0,
    )

    return tuple(
        float(value)
        for value in window_bounds(window, transform)
    )


def _empty_geodataframe(crs: CRS | str | None) -> gpd.GeoDataFrame:
    """Create an empty GeoDataFrame with a geometry column and CRS."""
    return gpd.GeoDataFrame(
        {"geometry": gpd.GeoSeries([], crs=crs)},
        geometry="geometry",
        crs=crs,
    )


def _validate_parameters(
    *,
    neighbourhood_radius_m: float,
    point_buffer_m: float,
    line_buffer_m: float,
    polygon_buffer_m: float,
    requests_timeout: int,
) -> None:
    parameters = {
        "Neighbourhood radius": neighbourhood_radius_m,
        "Point buffer": point_buffer_m,
        "Line buffer": line_buffer_m,
        "Polygon buffer": polygon_buffer_m,
    }

    for name, value in parameters.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        if value < 0:
            raise ValueError(f"{name} cannot be negative.")

    if requests_timeout <= 0:
        raise ValueError("Request timeout must be greater than zero.")