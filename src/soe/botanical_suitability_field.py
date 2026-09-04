#botanical_suitability_field.py
"""
Generate Rivelero's botanical suitability field from Sentinel-2 NDVI.

The result is a georeferenced scalar surface:

    (x, y) -> z = botanical suitability

where:
    x, y = projected map position on the DEM grid
    z    = botanical suitability in the interval 0..1

The field is aligned exactly to the supplied DEM so that it can later be
combined cell-by-cell with ``visibility_field.py``.

This module retains the original Copernicus Data Space Ecosystem Process API
NDVI retrieval, but adds:

- a field result dataclass;
- conversion of NDVI into a bounded 0..1 suitability score;
- reprojection onto the DEM grid;
- target-region masking;
- X/Y/Z mesh generation;
- GeoTIFF export;
- GUI-compatible display output.

Important scientific limitation
-------------------------------
The default conversion treats greener vegetation as more botanically suitable.
That is a generic vegetation-relevance proxy, not a calibrated probability that
a particular species is present. Supply ``suitability_function`` when a
species-specific response curve becomes available.

Requirements
------------
    numpy
    requests
    rasterio
    affine
    pyproj
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.warp import reproject, transform_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds

from .EXR_to_NDVI import UNREAL_LOCAL_CRS, load_ndvi_exr


CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)
CDSE_PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"

Geometry = Mapping[str, Any]
SuitabilityFunction = Callable[[np.ndarray], np.ndarray]

__all__ = [
    "BotanicalSuitabilityFieldResult",
    "NDVI",
    "build_botanical_suitability_field",
    "fetch_ndvi",
    "save_botanical_suitability_field",
]


@dataclass(slots=True)
class BotanicalSuitabilityFieldResult:
    """
    A georeferenced 3D botanical suitability surface.

    ``field`` is a 2D raster whose value is the field height z.

    Values:
        0.0 = botanically unsuitable according to the chosen NDVI response
        1.0 = maximally suitable according to the chosen NDVI response
        NaN = outside the target region, invalid DEM, or unavailable imagery

    The affine transform supplies the projected x and y position of every cell.
    """

    field: np.ndarray
    ndvi: np.ndarray
    valid_mask: np.ndarray
    target_mask: np.ndarray

    transform: Affine
    crs: CRS

    time_from: str | None
    time_to: str | None
    pixel_size_m: int
    max_cloud_coverage: int
    mosaicking_order: str

    source: str

    ndvi_floor: float
    ndvi_ceiling: float

    @property
    def z(self) -> np.ndarray:
        """Return the botanical-suitability field heights."""
        return self.field


    def surface_grids(
        self,
        max_rows: int = 400,
        max_columns: int = 400,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return downsampled X, Y and Z grids for a 3D surface plot.

        The original suitability field remains at full resolution.
        Display values are produced using average resampling.
        """
        source_height, source_width = self.field.shape

        # Preserve the raster's aspect ratio while fitting it within the
        # requested maximum display dimensions.
        display_scale = min(
            1.0,
            max_rows / source_height,
            max_columns / source_width,
        )

        display_height = max(
            1,
            round(source_height * display_scale),
        )
        display_width = max(
            1,
            round(source_width * display_scale),
        )

        # Convert masked values to NaN so they are excluded from averaging.
        source = np.ma.asarray(
            self.field,
            dtype=np.float32,
        ).filled(np.nan)

        display_field = np.full(
            (display_height, display_width),
            np.nan,
            dtype=np.float32,
        )

        # Make the new, larger display cells cover exactly the same
        # geographic area as the original raster.
        display_transform = self.transform * Affine.scale(
            source_width / display_width,
            source_height / display_height,
        )

        reproject(
            source=source,
            destination=display_field,
            src_transform=self.transform,
            src_crs=self.crs,
            src_nodata=np.nan,
            dst_transform=display_transform,
            dst_crs=self.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

        # Coordinates of the centres of the resampled display cells.
        columns, rows = np.meshgrid(
            np.arange(display_width, dtype=np.float64) + 0.5,
            np.arange(display_height, dtype=np.float64) + 0.5,
        )

        x = (
            display_transform.a * columns
            + display_transform.b * rows
            + display_transform.c
        )
        y = (
            display_transform.d * columns
            + display_transform.e * rows
            + display_transform.f
        )

        return x, y, np.ma.masked_invalid(display_field)

    def as_display_mapping(self) -> dict[str, Any]:
        """
        Return a mapping accepted by the GUI's ``coerce_raster_result()``.
        """
        return {
            "data": self.ndvi,
            "transform": self.transform,
            "crs": self.crs,
            "title": "NDVI",
            "colour_map": "RdYlGn",
            "colourbar_label": "NDVI",
            "vmin": -1.0,
            "vmax": 1.0,
        }


def build_botanical_suitability_field(
    *,
    dem_path: str | Path,
    target_geometry: Geometry | Sequence[Geometry] | None,
    time_from: str | None = None,
    time_to: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
    ndvi_floor: float = 0.0,
    ndvi_ceiling: float = 0.8,
    suitability_function: SuitabilityFunction | None = None,
    mask_water: bool = True,
    all_touched: bool = False,
    request_padding_cells: int = 1,
    ndvi_exr_path: str | Path | None = None,
    exr_capture_x_cm: float | None = None,
    exr_capture_y_cm: float | None = None,
    exr_ortho_width_cm: float | None = None,
    exr_encoded: bool = True,
) -> BotanicalSuitabilityFieldResult:
    """
    Build a DEM-aligned botanical suitability field.

    Parameters
    ----------
    dem_path:
        DEM GeoTIFF defining the field's output grid, transform and CRS.

    target_geometry:
        Polygon or polygons delimiting where the botanical target may occur.
        Coordinates must use the DEM's CRS. Pass ``None`` to use the complete
        valid DEM extent.

    time_from, time_to:
        ISO UTC timestamps accepted by the CDSE Process API. They are not
        required when ``ndvi_exr_path`` is supplied.

    ndvi_exr_path:
        Optional orthographic Unreal NDVI capture. When supplied, this source
        replaces the Sentinel-2 API request for this run.

    exr_capture_x_cm, exr_capture_y_cm, exr_ortho_width_cm:
        Unreal camera position and orthographic width used to georeference the
        EXR in the Unreal local coordinate system.

    exr_encoded:
        Decode the EXR using ``NDVI = 2 * encoded - 1``.

    client_id, client_secret:
        CDSE OAuth credentials. If omitted, the function reads
        ``CDSE_CLIENT_ID`` and ``CDSE_CLIENT_SECRET`` from the environment.

    pixel_size_m:
        Resolution requested from Sentinel-2. Ten metres matches B04/B08.

    max_cloud_coverage:
        Tile-level cloud-coverage filter from 0 to 100.

    mosaicking_order:
        CDSE/Sentinel Hub mosaicking order, normally ``leastCC`` or
        ``mostRecent``.

    ndvi_floor, ndvi_ceiling:
        Default linear suitability response:

            NDVI <= floor   -> suitability 0
            NDVI >= ceiling -> suitability 1

        Values between them are linearly interpolated.

    suitability_function:
        Optional custom callable receiving the aligned NDVI array and returning
        an array of suitability values. Its output is clipped to 0..1. Use this
        later for a species-specific ecological response.

    mask_water:
        Whether Sentinel-2 SCL water pixels should be treated as unavailable.
        This preserves the behaviour of the original NDVI module. Set False to
        retain water NDVI, which the default response normally maps near zero.

    all_touched:
        Include every DEM cell touched by the target polygon.

    request_padding_cells:
        Number of DEM cells added around the target-region request bounds.

    Returns
    -------
    BotanicalSuitabilityFieldResult
        A 0..1 field aligned exactly to the DEM raster grid.
    """
    using_exr = ndvi_exr_path is not None

    _validate_field_parameters(
        time_from=None if using_exr else time_from,
        time_to=None if using_exr else time_to,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
        ndvi_floor=ndvi_floor,
        ndvi_ceiling=ndvi_ceiling,
        request_padding_cells=request_padding_cells,
    )

    dem_path = Path(dem_path)

    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM does not exist: {dem_path}")

    with rasterio.open(dem_path) as source:
        if source.count < 1:
            raise ValueError("The DEM contains no raster bands.")

        if source.crs is None:
            raise ValueError("The DEM has no coordinate reference system.")

        if using_exr and source.crs != UNREAL_LOCAL_CRS:
            raise ValueError(
                "An Unreal NDVI EXR can only be aligned to a DEM using the "
                "same Unreal local CRS."
            )

        if not using_exr and not source.crs.is_projected:
            raise ValueError(
                "Use a projected DEM so the botanical and visibility fields "
                "share meaningful projected X/Y coordinates."
            )

        dem = source.read(1, masked=True).astype(np.float64)
        dem_shape = (source.height, source.width)
        dem_transform = source.transform
        dem_crs = source.crs

    valid_dem = ~np.ma.getmaskarray(dem)
    valid_dem &= np.isfinite(np.asarray(dem.filled(np.nan)))

    if not valid_dem.any():
        raise ValueError("The DEM contains no valid cells.")

    target_mask = _geometry_to_mask(
        target_geometry,
        shape=dem_shape,
        transform=dem_transform,
        all_touched=all_touched,
    )
    target_mask &= valid_dem

    if not target_mask.any():
        raise ValueError(
            "The target region does not contain any valid DEM cells."
        )

    if using_exr:
        missing_exr_metadata = [
            name
            for name, value in (
                ("exr_capture_x_cm", exr_capture_x_cm),
                ("exr_capture_y_cm", exr_capture_y_cm),
                ("exr_ortho_width_cm", exr_ortho_width_cm),
            )
            if value is None
        ]
        if missing_exr_metadata:
            raise ValueError(
                "An EXR NDVI source requires "
                + ", ".join(missing_exr_metadata)
                + "."
            )

        ndvi_result = load_ndvi_exr(
            ndvi_exr_path,
            capture_x_cm=float(exr_capture_x_cm),
            capture_y_cm=float(exr_capture_y_cm),
            ortho_width_cm=float(exr_ortho_width_cm),
            encoded=exr_encoded,
        )
        source_name = "Unreal Engine NDVI EXR"
    else:
        request_bounds_dem = _mask_bounds(
            target_mask,
            transform=dem_transform,
            padding_cells=request_padding_cells,
        )

        bbox_lonlat = transform_bounds(
            dem_crs,
            "EPSG:4326",
            *request_bounds_dem,
            densify_pts=21,
        )

        ndvi_result = fetch_ndvi(
            bbox_lonlat=bbox_lonlat,
            time_from=time_from,
            time_to=time_to,
            client_id=client_id,
            client_secret=client_secret,
            pixel_size_m=pixel_size_m,
            max_cloud_coverage=max_cloud_coverage,
            mosaicking_order=mosaicking_order,
            timeout=timeout,
            mask_water=mask_water,
        )
        source_name = "Sentinel-2 L2A NDVI"

    aligned_ndvi, aligned_valid = _align_ndvi_to_grid(
        ndvi=ndvi_result["ndvi"],
        valid_mask=ndvi_result["valid_mask"],
        source_transform=ndvi_result["transform"],
        source_crs=ndvi_result["crs"],
        destination_shape=dem_shape,
        destination_transform=dem_transform,
        destination_crs=dem_crs,
    )

    if suitability_function is None:
        suitability = _linear_ndvi_suitability(
            aligned_ndvi,
            ndvi_floor=ndvi_floor,
            ndvi_ceiling=ndvi_ceiling,
        )
    else:
        suitability = np.asarray(
            suitability_function(aligned_ndvi.copy()),
            dtype=np.float64,
        )

        if suitability.shape != dem_shape:
            raise ValueError(
                "The custom suitability function must return an array with "
                f"shape {dem_shape}, not {suitability.shape}."
            )

        suitability = np.clip(suitability, 0.0, 1.0)

    field_valid = target_mask & aligned_valid & np.isfinite(suitability)

    field = np.full(dem_shape, np.nan, dtype=np.float32)
    field[field_valid] = suitability[field_valid].astype(np.float32)

    aligned_ndvi = aligned_ndvi.astype(np.float32)
    aligned_ndvi[~(target_mask & aligned_valid)] = np.nan

    return BotanicalSuitabilityFieldResult(
        field=field,
        ndvi=aligned_ndvi,
        valid_mask=field_valid,
        target_mask=target_mask,
        transform=dem_transform,
        crs=dem_crs,
        time_from=time_from,
        time_to=time_to,
        pixel_size_m=int(pixel_size_m),
        max_cloud_coverage=int(max_cloud_coverage),
        mosaicking_order=mosaicking_order,
        source=source_name,  
        ndvi_floor=float(ndvi_floor),
        ndvi_ceiling=float(ndvi_ceiling),
    )


def save_botanical_suitability_field(
    result: BotanicalSuitabilityFieldResult,
    output_path: str | Path,
    *,
    nodata: float = -9999.0,
) -> Path:
    """Save the field height z as a georeferenced Float32 GeoTIFF."""
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
        destination.set_band_description(
            1,
            "botanical_suitability_field_height",
        )
        destination.update_tags(
            time_from=result.time_from or "",
            time_to=result.time_to or "",
            source=result.source,
            pixel_size_m=result.pixel_size_m,
            max_cloud_coverage=result.max_cloud_coverage,
            mosaicking_order=result.mosaicking_order,
            ndvi_floor=result.ndvi_floor,
            ndvi_ceiling=result.ndvi_ceiling,
            field_min=0.0,
            field_max=1.0,
        )

    return output_path


def fetch_ndvi(
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
    mask_water: bool = True,
) -> dict[str, Any]:
    """
    Retrieve Sentinel-2 L2A NDVI from the CDSE Process API.

    This is the lower-level retrieval function retained from the original
    module. Most Rivelero code should call
    ``build_botanical_suitability_field`` instead.
    """
    _validate_fetch_parameters(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
    )

    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    client_id = client_id or os.getenv("CDSE_CLIENT_ID")
    client_secret = client_secret or os.getenv("CDSE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Missing credentials. Pass client_id/client_secret or set "
            "CDSE_CLIENT_ID and CDSE_CLIENT_SECRET."
        )

    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0

    epsg = _utm_epsg_for_lonlat(center_lon, center_lat)
    bbox_projected = _project_bbox_crs84_to_epsg(
        bbox_lonlat,
        destination_epsg=epsg,
    )

    min_x, min_y, max_x, max_y = bbox_projected

    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Projected request bounds have non-positive size.")

    access_token = _get_cdse_access_token(
        client_id,
        client_secret,
        timeout=min(timeout, 30),
    )

    bad_scl_classes = [3, 7, 8, 9, 10, 11]

    if mask_water:
        bad_scl_classes.append(6)

    bad_scl_javascript = json.dumps(bad_scl_classes)

    evalscript = f"""
    //VERSION=3

    function setup() {{
        return {{
            input: ["B04", "B08", "SCL", "dataMask"],
            output: {{
                bands: 2,
                sampleType: "FLOAT32"
            }}
        }};
    }}

    function evaluatePixel(sample) {{
        let ndvi = index(sample.B08, sample.B04);
        let badSCL = {bad_scl_javascript}.includes(sample.SCL);
        let valid = (sample.dataMask === 1) && !badSCL;

        return [ndvi, valid ? 1 : 0];
    }}
    """

    request_body = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": (
                        "http://www.opengis.net/def/crs/EPSG/0/"
                        f"{epsg}"
                    )
                },
                "bbox": [min_x, min_y, max_x, max_y],
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": time_from,
                            "to": time_to,
                        },
                        "maxCloudCoverage": max_cloud_coverage,
                        "mosaickingOrder": mosaicking_order,
                    },
                    "processing": {
                        "upsampling": "NEAREST",
                        "downsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "resx": pixel_size_m,
            "resy": pixel_size_m,
        },
        "evalscript": evalscript,
    }

    response = requests.post(
        CDSE_PROCESS_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "image/tiff",
        },
        json=request_body,
        timeout=timeout,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            "CDSE Process API request failed with status "
            f"{response.status_code}: {response.text[:2000]}"
        )

    with MemoryFile(response.content) as memory_file:
        with memory_file.open() as dataset:
            data = dataset.read()
            profile = dataset.profile.copy()
            transform = dataset.transform
            crs = dataset.crs

    if data.shape[0] < 2:
        raise RuntimeError(
            "CDSE response did not contain both NDVI and validity bands."
        )

    ndvi = data[0].astype(np.float32)
    valid_mask = data[1] > 0.5
    ndvi[~valid_mask] = np.nan

    finite = np.isfinite(ndvi)

    if finite.any():
        statistics = {
            "mean_ndvi": float(np.nanmean(ndvi)),
            "median_ndvi": float(np.nanmedian(ndvi)),
            "min_ndvi": float(np.nanmin(ndvi)),
            "max_ndvi": float(np.nanmax(ndvi)),
        }
    else:
        statistics = {
            "mean_ndvi": float("nan"),
            "median_ndvi": float("nan"),
            "min_ndvi": float("nan"),
            "max_ndvi": float("nan"),
        }

    return {
        "ndvi": ndvi,
        "ndvi_array": ndvi,
        "valid_mask": valid_mask,
        "transform": transform,
        "raster_transform": transform,
        "crs": crs,
        "raster_crs": crs,
        "bounds_projected": bbox_projected,
        "raster_bounds": bbox_projected,
        "bbox_lonlat": bbox_lonlat,
        "epsg": epsg,
        "pixel_size_m": pixel_size_m,
        "profile": profile,
        **statistics,
    }


def NDVI(
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    pixel_size_m: int = 10,
    max_cloud_coverage: int = 20,
    mosaicking_order: str = "leastCC",
    timeout: int = 120,
    mask_water: bool = True,
) -> dict[str, Any]:
    """
    Backwards-compatible alias for the original standalone NDVI entry point.
    """
    return fetch_ndvi(
        bbox_lonlat=bbox_lonlat,
        time_from=time_from,
        time_to=time_to,
        client_id=client_id,
        client_secret=client_secret,
        pixel_size_m=pixel_size_m,
        max_cloud_coverage=max_cloud_coverage,
        mosaicking_order=mosaicking_order,
        timeout=timeout,
        mask_water=mask_water,
    )


def _linear_ndvi_suitability(
    ndvi: np.ndarray,
    *,
    ndvi_floor: float,
    ndvi_ceiling: float,
) -> np.ndarray:
    """Map NDVI linearly onto 0..1 using explicit ecological parameters."""
    suitability = (
        np.asarray(ndvi, dtype=np.float64) - ndvi_floor
    ) / (ndvi_ceiling - ndvi_floor)

    return np.clip(suitability, 0.0, 1.0)


def _align_ndvi_to_grid(
    *,
    ndvi: np.ndarray,
    valid_mask: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    destination_shape: tuple[int, int],
    destination_transform: Affine,
    destination_crs: CRS,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject NDVI and its validity mask onto the DEM field grid."""
    if source_crs is None:
        raise ValueError("The downloaded NDVI raster has no CRS.")

    if ndvi.shape != valid_mask.shape:
        raise ValueError("NDVI and validity mask shapes do not match.")

    source_nodata = -9999.0
    destination_nodata = -9999.0

    source_data = np.where(
        valid_mask & np.isfinite(ndvi),
        ndvi,
        source_nodata,
    ).astype(np.float32)

    destination_data = np.full(
        destination_shape,
        destination_nodata,
        dtype=np.float32,
    )

    reproject(
        source=source_data,
        destination=destination_data,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=source_nodata,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=destination_nodata,
        resampling=Resampling.bilinear,
        init_dest_nodata=True,
    )

    destination_valid = np.zeros(destination_shape, dtype=np.uint8)

    reproject(
        source=valid_mask.astype(np.uint8),
        destination=destination_valid,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=0,
        dst_transform=destination_transform,
        dst_crs=destination_crs,
        dst_nodata=0,
        resampling=Resampling.nearest,
        init_dest_nodata=True,
    )

    aligned_valid = (
        (destination_valid > 0)
        & np.isfinite(destination_data)
        & (destination_data != destination_nodata)
    )

    aligned_ndvi = destination_data.astype(np.float64)
    aligned_ndvi[~aligned_valid] = np.nan

    return aligned_ndvi, aligned_valid


def _geometry_to_mask(
    geometry: Geometry | Sequence[Geometry] | None,
    *,
    shape: tuple[int, int],
    transform: Affine,
    all_touched: bool,
) -> np.ndarray:
    """Rasterise one or more target polygons into a Boolean inclusion mask."""
    if geometry is None:
        return np.ones(shape, dtype=bool)

    geometries = _normalise_geometries(geometry)

    if not geometries:
        raise ValueError("The supplied target region has no geometries.")

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
    """Accept a geometry, Feature, FeatureCollection, or sequence."""
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


def _mask_bounds(
    mask: np.ndarray,
    *,
    transform: Affine,
    padding_cells: int,
) -> tuple[float, float, float, float]:
    """Return projected bounds around all True cells in a mask."""
    rows, columns = np.where(mask)

    if rows.size == 0:
        raise ValueError("Cannot calculate bounds of an empty mask.")

    row0 = max(0, int(rows.min()) - padding_cells)
    row1 = min(mask.shape[0], int(rows.max()) + 1 + padding_cells)
    column0 = max(0, int(columns.min()) - padding_cells)
    column1 = min(
        mask.shape[1],
        int(columns.max()) + 1 + padding_cells,
    )

    window = Window(
        col_off=column0,
        row_off=row0,
        width=column1 - column0,
        height=row1 - row0,
    )

    return window_bounds(window, transform)


def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """Select a local WGS84 UTM EPSG code."""
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(zone, 60))

    return (32600 if lat >= 0 else 32700) + zone


def _project_bbox_crs84_to_epsg(
    bbox_lonlat: tuple[float, float, float, float],
    destination_epsg: int,
) -> tuple[float, float, float, float]:
    """Project a longitude/latitude bounding box into a projected CRS."""
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{destination_epsg}",
        always_xy=True,
    )

    corners = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]

    projected = [
        transformer.transform(lon, lat)
        for lon, lat in corners
    ]

    x_values, y_values = zip(*projected, strict=True)

    return (
        min(x_values),
        min(y_values),
        max(x_values),
        max(y_values),
    )


def _get_cdse_access_token(
    client_id: str,
    client_secret: str,
    timeout: int,
) -> str:
    """Fetch a CDSE OAuth2 access token."""
    response = requests.post(
        CDSE_TOKEN_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    access_token = payload.get("access_token")

    if not access_token:
        raise RuntimeError(
            f"No access_token in token response: {payload}"
        )

    return str(access_token)


def _validate_field_parameters(
    *,
    time_from: str | None,
    time_to: str | None,
    pixel_size_m: int,
    max_cloud_coverage: int,
    mosaicking_order: str,
    ndvi_floor: float,
    ndvi_ceiling: float,
    request_padding_cells: int,
) -> None:
    if (time_from is None) != (time_to is None):
        raise ValueError(
            "Sentinel-2 start and end dates must either both be supplied or "
            "both be omitted for an EXR source."
        )

    if time_from is not None and time_to is not None:
        _validate_fetch_parameters(
            bbox_lonlat=(-1.0, -1.0, 1.0, 1.0),
            time_from=time_from,
            time_to=time_to,
            pixel_size_m=pixel_size_m,
            max_cloud_coverage=max_cloud_coverage,
            mosaicking_order=mosaicking_order,
        )
    else:
        if pixel_size_m <= 0:
            raise ValueError("Pixel size must be greater than zero.")
        if not 0 <= max_cloud_coverage <= 100:
            raise ValueError(
                "Maximum cloud coverage must be between 0 and 100."
            )
        if mosaicking_order not in {"leastCC", "mostRecent"}:
            raise ValueError(
                "Mosaicking order should normally be 'leastCC' or "
                "'mostRecent'."
            )

    if not np.isfinite(ndvi_floor):
        raise ValueError("NDVI floor must be finite.")

    if not np.isfinite(ndvi_ceiling):
        raise ValueError("NDVI ceiling must be finite.")

    if ndvi_ceiling <= ndvi_floor:
        raise ValueError("NDVI ceiling must be greater than NDVI floor.")

    if request_padding_cells < 0:
        raise ValueError("Request padding cannot be negative.")


def _validate_fetch_parameters(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    time_from: str,
    time_to: str,
    pixel_size_m: int,
    max_cloud_coverage: int,
    mosaicking_order: str,
) -> None:
    min_lon, min_lat, max_lon, max_lat = bbox_lonlat

    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(
            "bbox_lonlat must be "
            "(min_lon, min_lat, max_lon, max_lat) with min < max."
        )

    if not time_from.strip() or not time_to.strip():
        raise ValueError("Both time_from and time_to are required.")

    if pixel_size_m <= 0:
        raise ValueError("Pixel size must be greater than zero.")

    if not 0 <= max_cloud_coverage <= 100:
        raise ValueError(
            "Maximum cloud coverage must be between 0 and 100."
        )

    if mosaicking_order not in {"leastCC", "mostRecent"}:
        raise ValueError(
            "Mosaicking order should normally be 'leastCC' or "
            "'mostRecent'."
        )
