"""Converts an Unreal SceneCapture2D SceneDepth EXR to a GeoTIFF DSM
for use in Rivelero.
"""

from pathlib import Path

import OpenEXR
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


#######PROPERTIES: Change as necessary.#######
SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DEPTH = SCRIPT_DIR / "RT_DSM_Depth.EXR"
OUTPUT_TIF = SCRIPT_DIR / "BiomeSampleLevel_DSM.tif"


# SceneCapture2D properties.
#
# COPY THESE VALUES EXACTLY FROM:
# SceneCapture2D -> Details -> Transform -> Location
#
# If the capture is centred on the original Landscape,
# X and Y should both be 0.

CAPTURE_X_CM = 0.0
CAPTURE_Y_CM = 0.0

# IMPORTANT:
# This must exactly match the SceneCapture2D world Z coordinate.
CAPTURE_Z_CM = 15119.0

# Orthographic width shown in the SceneCapture2D Details panel.
ORTHO_WIDTH_CM = 101700.0


# This script assumes the SceneCapture rotation is:
#
# Roll  = 0
# Pitch = -90
# Yaw   = 0
#
# i.e. looking vertically downward.


EXPECTED_WIDTH = 1017
EXPECTED_HEIGHT = 1017

UNREAL_LOCAL_COORDINATE_MODE = "unreal_local"

NODATA_VALUE = -9999.0

##############################################

print(f"Reading: {INPUT_DEPTH}")
print(f"Exists:  {INPUT_DEPTH.exists()}")

if not INPUT_DEPTH.exists():
    raise FileNotFoundError(
        f"Could not find EXR file:\n{INPUT_DEPTH}"
    )


with OpenEXR.File(
    str(INPUT_DEPTH),
    separate_channels=True,
) as exr:

    channels = exr.channels()

    print(f"EXR channels: {list(channels.keys())}")

    # Unreal's R32F SceneDepth export should normally give us
    # a red channel. Be slightly defensive in case the exporter
    # names a single-channel image differently.
    if "R" in channels:
        depth_channel_name = "R"

    elif "Y" in channels:
        depth_channel_name = "Y"

    elif len(channels) == 1:
        depth_channel_name = next(iter(channels))

    else:
        raise ValueError(
            "Could not identify the SceneDepth channel.\n"
            f"Available EXR channels: {list(channels.keys())}"
        )

    print(f"Using depth channel: {depth_channel_name}")

    depth_cm = np.asarray(
        channels[depth_channel_name].pixels,
        dtype=np.float32,
    )



print(f"EXR shape: {depth_cm.shape}")
print(f"EXR dtype: {depth_cm.dtype}")


if depth_cm.ndim != 2:
    raise ValueError(
        "Expected a single-channel 2D SceneDepth image, "
        f"but got shape {depth_cm.shape}"
    )


if depth_cm.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
    raise ValueError(
        f"Expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
        f"but got {depth_cm.shape[1]}x{depth_cm.shape[0]}"
    )



finite_depth = depth_cm[np.isfinite(depth_cm)]

if finite_depth.size == 0:
    raise ValueError(
        "The SceneDepth EXR contains no finite depth values."
    )


print(
    f"Raw depth range: "
    f"{finite_depth.min():.3f} -> "
    f"{finite_depth.max():.3f} Unreal units"
)



# Unreal uses:
#
#     +X = forward
#     +Y = right
#     +Z = up
#
# Epic documents these axis directions, and Unreal's default
# distance unit is centimetres.
#
# With our SceneCapture looking vertically down:
#
#     raw image TOP   -> +Unreal X
#     raw image RIGHT -> +Unreal Y
#
# Our GeoTIFF convention instead needs:
#
#     raster RIGHT -> +X
#     raster TOP   -> +Y
#
# So transpose the image and reverse both resulting axes.


depth_cm = np.flipud(
    np.fliplr(
        depth_cm.T
    )
)

# ============================================================
# CONVERT DEPTH -> WORLD SURFACE HEIGHT
# ============================================================

# SceneDepth is measured in Unreal world distance units.
#
# Unreal's default distance unit is centimetres.
#
# Because the orthographic SceneCapture points vertically downward:
#
#     surface_Z_cm = capture_Z_cm - depth_cm
#
# Then convert centimetres -> metres.


valid = (
    np.isfinite(depth_cm)
    & (depth_cm > 0.0)
)

if not np.any(valid):
    raise ValueError(
        "No valid positive SceneDepth pixels were found."
    )


elevation_m = np.full(
    depth_cm.shape,
    NODATA_VALUE,
    dtype=np.float32,
)


elevation_m[valid] = (
    CAPTURE_Z_CM - depth_cm[valid]
) / 100.0

valid_elevations = elevation_m[valid]

print(
    f"Surface elevation range: "
    f"{valid_elevations.min():.3f} m -> "
    f"{valid_elevations.max():.3f} m"
)


# ============================================================
# BUILD THE UNREAL-LOCAL GEOTRANSFORM
# ============================================================


CAPTURE_X_M = CAPTURE_X_CM / 100.0
CAPTURE_Y_M = CAPTURE_Y_CM / 100.0

ORTHO_WIDTH_M = ORTHO_WIDTH_CM / 100.0


# SceneCapture2D's Ortho Width describes the width of the
# orthographic view in Unreal world units.
#
# Our target is square (1017 x 1017), so its physical height
# is also 1017 m.

ORTHO_HEIGHT_M = (
    ORTHO_WIDTH_M
    * EXPECTED_HEIGHT
    / EXPECTED_WIDTH
)


PIXEL_SIZE_X_M = (
    ORTHO_WIDTH_M
    / EXPECTED_WIDTH
)

PIXEL_SIZE_Y_M = (
    ORTHO_HEIGHT_M
    / EXPECTED_HEIGHT
)


print(
    f"Pixel size: "
    f"{PIXEL_SIZE_X_M:.6f} m x "
    f"{PIXEL_SIZE_Y_M:.6f} m"
)


WEST_EDGE_M = (
    CAPTURE_X_M
    - ORTHO_WIDTH_M / 2.0
)

NORTH_EDGE_M = (
    CAPTURE_Y_M
    + ORTHO_HEIGHT_M / 2.0
)


transform = from_origin(
    WEST_EDGE_M,
    NORTH_EDGE_M,
    PIXEL_SIZE_X_M,
    PIXEL_SIZE_Y_M,
)


# ============================================================
# LOCAL UNREAL CRS
# ============================================================

# This is NOT a real geographic CRS.
#
# It simply tells GIS software:
#
#     X and Y are local coordinates
#     units are metres

local_crs = CRS.from_wkt(
    '''
    LOCAL_CS["Unreal Engine Local Coordinates",
        LOCAL_DATUM["Unreal Engine Local Datum",0],
        UNIT["metre",1],
        AXIS["X",EAST],
        AXIS["Y",NORTH]]
    '''
)


# ============================================================
# WRITE GEOTIFF
# ============================================================


# Rasterio is now used ONLY here, to write the GeoTIFF.
#
# It never attempts to open the EXR.
# ---

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    width=EXPECTED_WIDTH,
    height=EXPECTED_HEIGHT,
    count=1,
    dtype="float32",
    crs=local_crs,
    transform=transform,
    nodata=NODATA_VALUE,
    compress="deflate",
) as dst:

    dst.write(
        elevation_m.astype(np.float32),
        1,
    )

    dst.update_tags(
        source="Unreal Engine SceneCapture2D SceneDepth DSM",
        coordinate_mode=UNREAL_LOCAL_COORDINATE_MODE,
        capture_source="SceneDepth in R",
        projection_type="Orthographic",
        unreal_capture_x_cm=str(CAPTURE_X_CM),
        unreal_capture_y_cm=str(CAPTURE_Y_CM),
        unreal_capture_z_cm=str(CAPTURE_Z_CM),
        ortho_width_cm=str(ORTHO_WIDTH_CM),
        pixel_spacing_x_m=str(PIXEL_SIZE_X_M),
        pixel_spacing_y_m=str(PIXEL_SIZE_Y_M),
    )


print()
print(f"Written: {OUTPUT_TIF}")


# ============================================================
# VERIFY OUTPUT
# ============================================================


with rasterio.open(OUTPUT_TIF) as src:

    print()
    print("----- OUTPUT GEOTIFF -----")

    print(f"Size:       {src.width} x {src.height}")
    print(f"CRS:        {src.crs}")
    print(f"Transform:  {src.transform}")
    print(f"Bounds:     {src.bounds}")
    print(f"Resolution: {src.res}")

    dsm = src.read(1)

    valid_dsm = (
        np.isfinite(dsm)
        & (dsm != NODATA_VALUE)
    )

    if not np.any(valid_dsm):
        raise ValueError(
            "The written GeoTIFF contains no valid DSM pixels."
        )

    values = dsm[valid_dsm]

    print(
        f"DSM range:  "
        f"{values.min():.3f} m -> "
        f"{values.max():.3f} m"
    )

    print(
        f"Valid cells: "
        f"{np.count_nonzero(valid_dsm):,} / "
        f"{dsm.size:,}"
    )


print()
print("Conversion complete.")