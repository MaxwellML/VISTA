"""
Converts an Unreal SceneCapture2D Perspective SceneDepth EXR
to a GeoTIFF DSM for use in Rivelero.

IMPORTANT:
- This is for a Perspective SceneCapture2D, NOT Orthographic.
- It assumes the camera looks vertically downward.
- It back-projects each pixel into world XYZ, then rasterises
  those 3D points onto a regular XY grid.
- Because it keeps the maximum elevation in each output cell,
  the result is a DSM (surface model), not a bare-earth DEM.
"""

from pathlib import Path

import OpenEXR
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


# ============================================================
# PROPERTIES
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_DEPTH = SCRIPT_DIR / "RT_RuralAustralia_DepthTEST2.exr"
OUTPUT_TIF = SCRIPT_DIR / "RT_RuralAustralia_DepthTEST2.tif"

# ------------------------------------------------------------
# SceneCapture2D properties
# ------------------------------------------------------------
# COPY THESE FROM UNREAL

CAPTURE_X_CM = 0.0
CAPTURE_Y_CM = 0.0
CAPTURE_Z_CM = 51000.0

# Perspective capture FOV shown in SceneCapture2D Details.
# Unreal's SceneCapture2D FOV is treated here as horizontal FOV.
HORIZONTAL_FOV_DEG = 90.0

# This script assumes the camera is looking vertically downward.
#
# Your current actor is effectively:
#   Roll  = 0
#   Pitch = 270 degrees (equivalent to -90 degrees)
#   Yaw   = 0
#
# If you change the camera orientation, this script must be updated.

EXPECTED_WIDTH = 2048
EXPECTED_HEIGHT = 2048

# Output raster dimensions.
# Keeping 2048 x 2048 is sensible because your input is 2048 x 2048.
OUTPUT_WIDTH = 2048
OUTPUT_HEIGHT = 2048

UNREAL_LOCAL_COORDINATE_MODE = "unreal_local"

NODATA_VALUE = -9999.0

# Very small / zero depth values are treated as invalid.
MIN_VALID_DEPTH_CM = 1e-6

# ============================================================


print(f"Reading: {INPUT_DEPTH}")
print(f"Exists:  {INPUT_DEPTH.exists()}")

if not INPUT_DEPTH.exists():
    raise FileNotFoundError(
        f"Could not find EXR file:\n{INPUT_DEPTH}"
    )


# ============================================================
# READ EXR
# ============================================================

with OpenEXR.File(
    str(INPUT_DEPTH),
    separate_channels=True,
) as exr:

    channels = exr.channels()

    print(f"EXR channels: {list(channels.keys())}")

    # SceneDepth is expected in R.
    # Be a little defensive in case channel naming differs.
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


# ============================================================
# BUILD A PERSPECTIVE RAY FOR EVERY PIXEL
# ============================================================
#
# Unreal camera local axes:
#   +X = forward
#   +Y = right
#   +Z = up
#
# We treat the FOV as horizontal FOV.
# For each pixel we build a ray in camera-local space.

height, width = depth_cm.shape

aspect = width / height
horizontal_fov_rad = np.deg2rad(HORIZONTAL_FOV_DEG)

vertical_fov_rad = 2.0 * np.arctan(
    np.tan(horizontal_fov_rad / 2.0) / aspect
)

tan_half_hfov = np.tan(horizontal_fov_rad / 2.0)
tan_half_vfov = np.tan(vertical_fov_rad / 2.0)

# Pixel-centre coordinates in normalized image space:
#   u = -1 at left, +1 at right
#   v = +1 at top,  -1 at bottom
u = (
    (np.arange(width, dtype=np.float64) + 0.5)
    / width
    * 2.0
    - 1.0
)

v = (
    1.0
    - (np.arange(height, dtype=np.float64) + 0.5)
    / height
    * 2.0
)

u_grid, v_grid = np.meshgrid(u, v)

# Camera-local direction before normalization:
#
# [forward, right, up] =
# [1,
#  u * tan(hfov/2),
#  v * tan(vfov/2)]
ray_forward = np.ones_like(u_grid, dtype=np.float64)
ray_right = u_grid * tan_half_hfov
ray_up = v_grid * tan_half_vfov


# ============================================================
# CAMERA-LOCAL RAYS -> WORLD RAYS
# ============================================================
#
# This script assumes:
#   Roll  = 0
#   Pitch = -90 (or 270)
#   Yaw   = 0
#
# For that orientation:
#   camera local +forward -> world -Z
#   camera local +right   -> world +Y
#   camera local +up      -> world +X
#
# So:
ray_world_x = ray_up
ray_world_y = ray_right
ray_world_z = -ray_forward


# ============================================================
# BACK-PROJECT PERSPECTIVE SCENE DEPTH
# ============================================================
#
# Unreal SceneDepth here is view-space/forward depth,
# NOT Euclidean distance along a normalized viewing ray.
#
# For a perspective camera:
#
# camera-local point =
#
#   forward = depth
#   right   = depth * u * tan(HFOV / 2)
#   up      = depth * v * tan(VFOV / 2)
#
# Camera orientation:
#
#   Pitch = -90 / 270
#   Yaw   = 0
#   Roll  = 0
#
# Therefore:
#
#   camera forward -> world -Z
#   camera right   -> world +Y
#   camera up      -> world +X

valid = (
    np.isfinite(depth_cm)
    & (depth_cm > MIN_VALID_DEPTH_CM)
)

if not np.any(valid):
    raise ValueError(
        "No valid positive SceneDepth pixels were found."
    )

depth_cm_64 = depth_cm.astype(np.float64)

camera_right_offset_cm = (
    depth_cm_64 * u_grid * tan_half_hfov
)

camera_up_offset_cm = (
    depth_cm_64 * v_grid * tan_half_vfov
)

world_x_cm = np.full(
    depth_cm.shape,
    np.nan,
    dtype=np.float64,
)

world_y_cm = np.full(
    depth_cm.shape,
    np.nan,
    dtype=np.float64,
)

world_z_cm = np.full(
    depth_cm.shape,
    np.nan,
    dtype=np.float64,
)

world_x_cm[valid] = (
    CAPTURE_X_CM
    + camera_up_offset_cm[valid]
)

world_y_cm[valid] = (
    CAPTURE_Y_CM
    + camera_right_offset_cm[valid]
)

world_z_cm[valid] = (
    CAPTURE_Z_CM
    - depth_cm_64[valid]
)

world_x_m = world_x_cm / 100.0
world_y_m = world_y_cm / 100.0
elevation_m = world_z_cm / 100.0


# ============================================================
# RASTERISE BACK-PROJECTED POINTS TO A REGULAR XY GRID
# ============================================================
#
# Since this is a perspective image, pixels are NOT already on
# a regular ground grid.
#
# So we:
# 1. take every valid pixel's world X, world Y, world Z
# 2. define a regular XY raster extent
# 3. assign each sample to a raster cell
# 4. keep the maximum Z per cell (DSM behaviour)

x_vals = world_x_m[valid].ravel()
y_vals = world_y_m[valid].ravel()
z_vals = elevation_m[valid].ravel()

west = float(np.min(x_vals))
east = float(np.max(x_vals))
south = float(np.min(y_vals))
north = float(np.max(y_vals))

if not (east > west and north > south):
    raise ValueError(
        "Invalid projected bounds derived from the perspective capture."
    )

pixel_size_x_m = (east - west) / OUTPUT_WIDTH
pixel_size_y_m = (north - south) / OUTPUT_HEIGHT

print(
    f"Projected bounds: "
    f"W={west:.3f}, E={east:.3f}, "
    f"S={south:.3f}, N={north:.3f}"
)

print(
    f"Output pixel size: "
    f"{pixel_size_x_m:.6f} m x "
    f"{pixel_size_y_m:.6f} m"
)

# Map projected XY to raster row/col.
cols = np.floor((x_vals - west) / pixel_size_x_m).astype(np.int64)
rows = np.floor((north - y_vals) / pixel_size_y_m).astype(np.int64)

# Clip edge cases caused by points lying exactly on the max boundary.
cols = np.clip(cols, 0, OUTPUT_WIDTH - 1)
rows = np.clip(rows, 0, OUTPUT_HEIGHT - 1)

flat_idx = rows * OUTPUT_WIDTH + cols

# DSM raster: keep max elevation per cell.
dsm_flat = np.full(
    OUTPUT_WIDTH * OUTPUT_HEIGHT,
    -np.inf,
    dtype=np.float32,
)

np.maximum.at(
    dsm_flat,
    flat_idx,
    z_vals.astype(np.float32),
)

# Optional diagnostics: count how many samples land in each cell.
count_flat = np.zeros(
    OUTPUT_WIDTH * OUTPUT_HEIGHT,
    dtype=np.uint32,
)
np.add.at(count_flat, flat_idx, 1)

dsm = dsm_flat.reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH))
sample_count = count_flat.reshape((OUTPUT_HEIGHT, OUTPUT_WIDTH))

valid_cells = np.isfinite(dsm)

filled_cells = int(np.count_nonzero(valid_cells))
total_cells = int(dsm.size)

print(
    f"Rasterised cells with data: "
    f"{filled_cells:,} / {total_cells:,}"
)

# Convert empty cells to nodata.
dsm[~valid_cells] = NODATA_VALUE

if filled_cells == 0:
    raise ValueError(
        "The rasterised DSM contains no valid cells."
    )


# ============================================================
# BUILD GEOTRANSFORM
# ============================================================

transform = from_origin(
    west,
    north,
    pixel_size_x_m,
    pixel_size_y_m,
)

# ============================================================
# LOCAL UNREAL CRS
# ============================================================
#
# This is NOT a real geographic CRS.
# It simply tells GIS software:
#   X and Y are local coordinates
#   units are metres
#
# We interpret:
#   Unreal X -> Easting
#   Unreal Y -> Northing

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

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",
    width=OUTPUT_WIDTH,
    height=OUTPUT_HEIGHT,
    count=1,
    dtype="float32",
    crs=local_crs,
    transform=transform,
    nodata=NODATA_VALUE,
    compress="deflate",
) as dst:

    dst.write(
        dsm.astype(np.float32),
        1,
    )

    dst.update_tags(
        source="Unreal Engine SceneCapture2D Perspective SceneDepth DSM",
        coordinate_mode=UNREAL_LOCAL_COORDINATE_MODE,
        capture_source="SceneDepth in R",
        projection_type="Perspective",
        horizontal_fov_deg=str(HORIZONTAL_FOV_DEG),
        unreal_capture_x_cm=str(CAPTURE_X_CM),
        unreal_capture_y_cm=str(CAPTURE_Y_CM),
        unreal_capture_z_cm=str(CAPTURE_Z_CM),
        output_pixel_size_x_m=str(pixel_size_x_m),
        output_pixel_size_y_m=str(pixel_size_y_m),
        assumed_rotation="Roll=0, Pitch=270(-90), Yaw=0",
        rasterisation_method="max_elevation_per_cell",
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

    out = src.read(1)

    valid_out = (
        np.isfinite(out)
        & (out != NODATA_VALUE)
    )

    if not np.any(valid_out):
        raise ValueError(
            "The written GeoTIFF contains no valid DSM pixels."
        )

    values = out[valid_out]

    print(
        f"DSM range:  "
        f"{values.min():.3f} m -> "
        f"{values.max():.3f} m"
    )

    print(
        f"Valid cells: "
        f"{np.count_nonzero(valid_out):,} / "
        f"{out.size:,}"
    )

print()
print("Conversion complete.")