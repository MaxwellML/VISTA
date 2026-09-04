#dem_downsampler.py
"""Resolution limiting for temporary DEM/DSM processing rasters."""

from __future__ import annotations

from math import ceil, hypot, isclose, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject


DEFAULT_WORKING_RESOLUTION_METRES = 1.0


def pixel_resolution(transform: Affine) -> tuple[float, float]:
    """Return pixel-vector lengths in the raster CRS units."""

    x_resolution = hypot(transform.a, transform.d)
    y_resolution = hypot(transform.b, transform.e)

    if (
        not isfinite(x_resolution)
        or not isfinite(y_resolution)
        or x_resolution <= 0.0
        or y_resolution <= 0.0
    ):
        raise ValueError("The raster has an invalid affine pixel resolution.")

    return x_resolution, y_resolution


def _require_projected_metre_crs(crs: object | None) -> None:
    """Require a projected CRS whose horizontal units are metres."""

    if crs is None or not crs.is_projected:
        raise ValueError(
            "DEM/DSM downsampling to a metre resolution requires a "
            "projected CRS. Reproject the raster to a metre-based CRS first."
        )

    try:
        unit_name, unit_factor = crs.linear_units_factor
    except (AttributeError, ValueError) as error:
        raise ValueError(
            "Could not determine the projected CRS linear units. Reproject "
            "the raster to a metre-based CRS before processing."
        ) from error

    if not isclose(float(unit_factor), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "DEM/DSM downsampling requires projected CRS units in metres; "
            f"the raster uses {unit_name!r}."
        )


def _copy_tags(source: object, destination: object) -> None:
    """Copy dataset and per-band tags, including named namespaces."""

    destination.update_tags(**source.tags())
    for namespace in source.tag_namespaces():
        if namespace:
            tags = source.tags(ns=namespace)
            if tags:
                destination.update_tags(ns=namespace, **tags)

    for band_index in source.indexes:
        destination.update_tags(band_index, **source.tags(band_index))
        for namespace in source.tag_namespaces(band_index):
            if namespace:
                tags = source.tags(band_index, ns=namespace)
                if tags:
                    destination.update_tags(
                        band_index,
                        ns=namespace,
                        **tags,
                    )


def copy_raster_metadata(source: object, destination: object) -> None:
    """Copy metadata not completely represented by a rasterio profile."""

    _copy_tags(source, destination)

    for band_index, description in zip(
        source.indexes,
        source.descriptions,
        strict=True,
    ):
        if description is not None:
            destination.set_band_description(band_index, description)

    for band_index, unit in zip(
        source.indexes,
        source.units,
        strict=True,
    ):
        if unit is not None:
            destination.set_band_unit(band_index, unit)

    destination.scales = source.scales
    destination.offsets = source.offsets
    destination.colorinterp = source.colorinterp


def downsample_raster_for_processing(
    path: str | Path,
    *,
    target_resolution_m: float = DEFAULT_WORKING_RESOLUTION_METRES,
) -> bool:
    """Limit a processing raster to no finer than the requested resolution.

    The raster is replaced atomically only when at least one pixel dimension
    is finer than ``target_resolution_m``. Maximum-value resampling preserves
    DSM terrain, building, and vegetation peaks used by visibility analysis.

    Returns ``True`` when the file was resampled and ``False`` when it was
    already at or below the processing cost limit.
    """

    path = Path(path)
    if target_resolution_m <= 0.0:
        raise ValueError("target_resolution_m must be greater than zero.")

    temporary_path: Path | None = None

    try:
        with rasterio.open(path) as source:
            x_resolution, y_resolution = pixel_resolution(source.transform)
            x_is_finer = x_resolution < target_resolution_m
            y_is_finer = y_resolution < target_resolution_m

            if not (x_is_finer or y_is_finer):
                return False

            _require_projected_metre_crs(source.crs)

            # Never increase the number of samples on an already-coarser axis.
            # Normal square-pixel DEMs/DSMs therefore become exactly 1 m x 1 m;
            # a mixed-resolution raster retains any axis already coarser than
            # the target instead of being upsampled.
            output_x_resolution = max(
                x_resolution,
                target_resolution_m,
            )
            output_y_resolution = max(
                y_resolution,
                target_resolution_m,
            )

            width = max(
                1,
                ceil(
                    source.width
                    * x_resolution
                    / output_x_resolution
                ),
            )
            height = max(
                1,
                ceil(
                    source.height
                    * y_resolution
                    / output_y_resolution
                ),
            )

            destination_transform = source.transform * Affine.scale(
                output_x_resolution / x_resolution,
                output_y_resolution / y_resolution,
            )

            profile = source.profile.copy()
            profile.update(
                width=width,
                height=height,
                transform=destination_transform,
                crs=source.crs,
                nodata=source.nodata,
            )

            with NamedTemporaryFile(
                prefix=f".{path.stem}-",
                suffix=path.suffix,
                dir=path.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            with rasterio.open(temporary_path, "w", **profile) as destination:
                for band_index in source.indexes:
                    reproject(
                        source=rasterio.band(source, band_index),
                        destination=rasterio.band(destination, band_index),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodatavals[band_index - 1],
                        dst_transform=destination_transform,
                        dst_crs=source.crs,
                        dst_nodata=source.nodatavals[band_index - 1],
                        resampling=Resampling.max,
                        init_dest_nodata=True,
                    )

                copy_raster_metadata(source, destination)

        temporary_path.replace(path)
        temporary_path = None
        return True
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
