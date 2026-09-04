# EXR_to_NDVI.py
"""Load and georeference orthographic Unreal Engine NDVI EXR captures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin


# ============================================================
# UNREAL PROJECTED CRS
# ============================================================
#
# This MUST match the synthetic projected CRS used by the
# orthographic DSM converter.
#
# The scene is still an Unreal-local coordinate system.
# This Transverse Mercator definition simply gives that local
# coordinate space a valid projected metre-based CRS so Rasterio,
# GDAL and Rivelero can operate on it.
#
# Coordinate convention:
#
#   raster Easting  = Unreal Y
#   raster Northing = Unreal X
#
# ============================================================

UNREAL_LOCAL_CRS = CRS.from_proj4(
    "+proj=tmerc "
    "+lat_0=0 "
    "+lon_0=0 "
    "+k=1 "
    "+x_0=0 "
    "+y_0=0 "
    "+datum=WGS84 "
    "+units=m "
    "+no_defs"
)


if not UNREAL_LOCAL_CRS.is_projected:
    raise RuntimeError(
        "The Unreal coordinate system must be recognised as projected."
    )


def load_ndvi_exr(
    path: str | Path,
    *,
    capture_x_cm: float,
    capture_y_cm: float,
    ortho_width_cm: float,
    encoded: bool = True,
) -> dict[str, Any]:
    """Load an orthographic Unreal NDVI EXR and attach spatial metadata.

    Camera assumptions:
        Pitch = -90 degrees
        Yaw   = 0 degrees
        Roll  = 0 degrees

    Raster coordinate convention:
        columns -> Unreal +Y -> raster Easting
        rows    -> Unreal -X -> raster South

    Therefore:
        raster Easting  = Unreal Y
        raster Northing = Unreal X

    This is deliberately the same coordinate convention used by the
    orthographic DSM converter.

    If ``encoded=True``, EXR values are decoded using:

        NDVI = 2 * encoded - 1

    Zero-valued source pixels are treated as capture background.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"NDVI EXR does not exist: {path}"
        )

    if (
        not np.isfinite(ortho_width_cm)
        or ortho_width_cm <= 0
    ):
        raise ValueError(
            "EXR orthographic width must be greater than zero."
        )

    if (
        not np.isfinite(capture_x_cm)
        or not np.isfinite(capture_y_cm)
    ):
        raise ValueError(
            "EXR capture X and Y must be finite."
        )


    # ========================================================
    # READ EXR
    # ========================================================

    try:
        import OpenEXR

    except ImportError as error:
        raise RuntimeError(
            "Reading NDVI EXR files requires "
            "the OpenEXR Python package."
        ) from error


    with OpenEXR.File(
        str(path),
        separate_channels=True,
    ) as exr:

        channels = exr.channels()

        if "R" in channels:
            channel_name = "R"

        elif "Y" in channels:
            channel_name = "Y"

        elif len(channels) == 1:
            channel_name = next(iter(channels))

        else:
            raise ValueError(
                "Could not identify the NDVI channel in the EXR. "
                f"Available channels: {list(channels)}"
            )

        data = np.asarray(
            channels[channel_name].pixels,
            dtype=np.float32,
        )


    # ========================================================
    # VALIDATE EXR
    # ========================================================

    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D EXR channel, got {data.shape}."
        )


    # ========================================================
    # DECODE NDVI
    # ========================================================

    valid_mask = (
        np.isfinite(data)
        & (data != 0.0)
    )


    if encoded:
        ndvi = (
            2.0 * data - 1.0
        )
    else:
        ndvi = data.copy()


    # IMPORTANT:
    #
    # Do NOT transpose or flip the image.
    #
    # The raw orthographic SceneCapture orientation is already:
    #
    #   columns left -> right = Unreal +Y
    #   rows top -> bottom    = Unreal -X
    #
    # The DSM converter uses:
    #
    #   raster Easting  = Unreal Y
    #   raster Northing = Unreal X
    #
    # which means the EXR already has the correct raster orientation.

    ndvi = ndvi.astype(
        np.float32,
        copy=False,
    )

    ndvi[~valid_mask] = np.nan


    # ========================================================
    # ORTHOGRAPHIC GEOMETRY
    # ========================================================

    height, width = data.shape

    aspect = width / height


    ortho_width_m = (
        ortho_width_cm
        / 100.0
    )

    ortho_height_m = (
        ortho_width_m
        / aspect
    )


    capture_x_m = (
        capture_x_cm
        / 100.0
    )

    capture_y_m = (
        capture_y_cm
        / 100.0
    )


    # ========================================================
    # GEOREFERENCE
    # ========================================================
    #
    # This exactly mirrors the DSM converter:
    #
    #   raster horizontal axis = Unreal Y
    #   raster vertical axis   = Unreal X
    #
    # Therefore:
    #
    #   west  = capture Unreal Y - half width
    #   north = capture Unreal X + half height
    #
    # ========================================================

    west = (
        capture_y_m
        - ortho_width_m / 2.0
    )

    north = (
        capture_x_m
        + ortho_height_m / 2.0
    )


    pixel_size_x_m = (
        ortho_width_m
        / width
    )

    pixel_size_y_m = (
        ortho_height_m
        / height
    )


    transform = from_origin(
        west,
        north,
        pixel_size_x_m,
        pixel_size_y_m,
    )


    # ========================================================
    # BOUNDS
    # ========================================================

    east = (
        west
        + width * pixel_size_x_m
    )

    south = (
        north
        - height * pixel_size_y_m
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "ndvi": ndvi,
        "valid_mask": valid_mask,

        "transform": transform,
        "crs": UNREAL_LOCAL_CRS,

        "pixel_size_x_m": pixel_size_x_m,
        "pixel_size_y_m": pixel_size_y_m,

        "bounds": {
            "west": west,
            "east": east,
            "north": north,
            "south": south,
        },
    }