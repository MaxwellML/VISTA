"""Load and georeference orthographic Unreal Engine NDVI EXR captures."""


from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin


UNREAL_LOCAL_CRS = CRS.from_wkt(
    '''
LOCAL_CS["Unreal Engine Local Coordinates",
LOCAL_DATUM["Unreal Engine Local Datum",0],
UNIT["metre",1],
AXIS["X",EAST],
AXIS["Y",NORTH]]
'''
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

    The camera is assumed to use pitch=-90, yaw=0 and roll=0. Unreal X maps
    to easting and Unreal Y maps to northing. With ``encoded=True``, channel
    values are decoded using ``NDVI = 2 * encoded - 1``. Zero-valued source
    pixels are treated as the capture background before decoding.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"NDVI EXR does not exist: {path}")

    if not np.isfinite(ortho_width_cm) or ortho_width_cm <= 0:
        raise ValueError("EXR orthographic width must be greater than zero.")

    if not np.isfinite(capture_x_cm) or not np.isfinite(capture_y_cm):
        raise ValueError("EXR capture X and Y must be finite.")

    try:
        import OpenEXR
    except ImportError as error:
        raise RuntimeError(
            "Reading NDVI EXR files requires the OpenEXR Python package."
        ) from error

    with OpenEXR.File(str(path), separate_channels=True) as exr:
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

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D EXR channel, got {data.shape}.")

    valid_mask = np.isfinite(data) & (data != 0.0)
    ndvi = 2.0 * data - 1.0 if encoded else data.copy()

    # Raw camera columns point along +Y and rows along -X. Reorient so raster
    # columns point east (+X) and raster rows point south (-Y).
    ndvi = np.flip(ndvi, axis=(0, 1)).T
    valid_mask = np.flip(valid_mask, axis=(0, 1)).T
    ndvi[~valid_mask] = np.nan

    original_height, original_width = data.shape
    ortho_width_m = ortho_width_cm / 100.0
    ortho_height_m = ortho_width_m * original_height / original_width
    capture_x_m = capture_x_cm / 100.0
    capture_y_m = capture_y_cm / 100.0

    west = capture_x_m - ortho_height_m / 2.0
    north = capture_y_m + ortho_width_m / 2.0
    pixel_size_x_m = ortho_height_m / ndvi.shape[1]
    pixel_size_y_m = ortho_width_m / ndvi.shape[0]

    transform = from_origin(
        west,
        north,
        pixel_size_x_m,
        pixel_size_y_m,
    )

    return {
        "ndvi": ndvi,
        "valid_mask": valid_mask,
        "transform": transform,
        "crs": UNREAL_LOCAL_CRS,
        "pixel_size_x_m": pixel_size_x_m,
        "pixel_size_y_m": pixel_size_y_m,
        "bounds": {
            "west": west,
            "east": west + ndvi.shape[1] * pixel_size_x_m,
            "north": north,
            "south": north - ndvi.shape[0] * pixel_size_y_m,
        },
    }
