"""
Converts an Unreal SceneCapture2D ORTHOGRAPHIC SceneDepth EXR
to a GeoTIFF DSM for use in Rivelero.

IMPORTANT:
- This is for an Orthographic SceneCapture2D, NOT Perspective.
- It assumes the camera looks vertically downward.
- It assumes Capture Source = SceneDepth in R.
- It assumes Unreal units are centimetres.
- It writes the orthographic capture directly to a regular raster grid.
- Because depth includes trees/road/etc, the output is a DSM
  (surface model), not a bare-earth DEM.

Assumed camera rotation:
    Roll  = 0
    Pitch = 270 degrees (equivalent to -90)
    Yaw   = 0

CRS:
- Unreal's local coordinates are not inherently geographic.
- A synthetic Transverse Mercator projected CRS is embedded into
  the GeoTIFF so downstream GIS code recognises the raster as
  projected and measured in metres.
- This does NOT mean the Unreal scene is actually located at
  latitude 0 / longitude 0.
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

INPUT_DEPTH = SCRIPT_DIR / "RT_RuralAustralia_Depth.EXR"
OUTPUT_TIF = SCRIPT_DIR / "RT_RuralAustralia_Depth_OrthoDSM.tif"


# ------------------------------------------------------------
# SceneCapture2D properties
# ------------------------------------------------------------
# COPY THESE FROM UNREAL

CAPTURE_X_CM = 0.0
CAPTURE_Y_CM = 0.0
CAPTURE_Z_CM = 51000.0

# Set this to the SceneCapture2D's Ortho Width from Unreal.
ORTHO_WIDTH_CM = 7000.0


# ------------------------------------------------------------
# Expected render target size
# ------------------------------------------------------------

EXPECTED_WIDTH = 2048
EXPECTED_HEIGHT = 2048


# ------------------------------------------------------------
# Depth handling
# ------------------------------------------------------------

NODATA_VALUE = -9999.0

# Very small / zero depth values are treated as invalid.
MIN_VALID_DEPTH_CM = 1e-6

# Set to a number if you want to reject very large depth values.
# Example:
# MAX_VALID_DEPTH_CM = 100000.0
MAX_VALID_DEPTH_CM = None


# ------------------------------------------------------------
# Metadata
# ------------------------------------------------------------

COORDINATE_MODE = "synthetic_projected_unreal_local"


# ============================================================
# READ EXR
# ============================================================

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

    # SceneDepth is expected in R.
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


# ============================================================
# VALIDATE EXR
# ============================================================

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
# VALID DEPTH PIXELS
# ============================================================

valid = (
    np.isfinite(depth_cm)
    & (depth_cm > MIN_VALID_DEPTH_CM)
)

if MAX_VALID_DEPTH_CM is not None:
    valid &= depth_cm < MAX_VALID_DEPTH_CM


if not np.any(valid):
    raise ValueError(
        "No valid positive SceneDepth pixels were found."
    )


print(
    f"Valid depth pixels: "
    f"{np.count_nonzero(valid):,} / "
    f"{depth_cm.size:,}"
)


# ============================================================
# ORTHOGRAPHIC GEOMETRY
# ============================================================

height, width = depth_cm.shape

aspect = width / height


# Unreal Ortho Width is the horizontal span of the camera view.
ortho_width_cm = float(ORTHO_WIDTH_CM)

ortho_height_cm = (
    ortho_width_cm / aspect
)


# Convert capture dimensions to metres.
ortho_width_m = (
    ortho_width_cm / 100.0
)

ortho_height_m = (
    ortho_height_cm / 100.0
)


# Each orthographic input pixel already corresponds to a fixed
# ground-grid location.
pixel_size_x_m = (
    ortho_width_m / width
)

pixel_size_y_m = (
    ortho_height_m / height
)


print()
print("----- ORTHOGRAPHIC CAPTURE -----")
print(f"Aspect ratio:      {aspect:.6f}")
print(f"Ortho width:       {ortho_width_m:.6f} m")
print(f"Ortho height:      {ortho_height_m:.6f} m")

print(
    f"Output pixel size: "
    f"{pixel_size_x_m:.6f} m x "
    f"{pixel_size_y_m:.6f} m"
)


# ============================================================
# DEPTH -> ELEVATION
# ============================================================
#
# For a vertically downward orthographic SceneCapture:
#
#     surface Z = camera Z - SceneDepth
#
# Unreal units are centimetres, so divide by 100 for metres.
# ============================================================

depth_cm_64 = depth_cm.astype(np.float64)


elevation_m = np.full(
    depth_cm.shape,
    NODATA_VALUE,
    dtype=np.float32,
)


elevation_m[valid] = (
    (
        CAPTURE_Z_CM
        - depth_cm_64[valid]
    )
    / 100.0
).astype(np.float32)


valid_out = (
    np.isfinite(elevation_m)
    & (elevation_m != NODATA_VALUE)
)


if not np.any(valid_out):
    raise ValueError(
        "The output DSM contains no valid pixels."
    )


values = elevation_m[valid_out]


print()
print(
    f"DSM elevation range: "
    f"{values.min():.3f} m -> "
    f"{values.max():.3f} m"
)


# ============================================================
# BUILD GEOTRANSFORM
# ============================================================
#
# Camera orientation:
#
#   camera local +forward -> world -Z
#   camera local +right   -> world +Y
#   camera local +up      -> world +X
#
# Therefore, for this camera rotation:
#
#   raster columns -> Unreal Y
#   raster rows    -> Unreal X
#
# We represent:
#
#   Unreal Y as raster Easting
#   Unreal X as raster Northing
# ============================================================


west_m = (
    CAPTURE_Y_CM
    - ortho_width_cm / 2.0
) / 100.0


east_m = (
    CAPTURE_Y_CM
    + ortho_width_cm / 2.0
) / 100.0


north_m = (
    CAPTURE_X_CM
    + ortho_height_cm / 2.0
) / 100.0


south_m = (
    CAPTURE_X_CM
    - ortho_height_cm / 2.0
) / 100.0


print()
print(
    f"Bounds: "
    f"W={west_m:.3f}, "
    f"E={east_m:.3f}, "
    f"S={south_m:.3f}, "
    f"N={north_m:.3f}"
)


transform = from_origin(
    west_m,
    north_m,
    pixel_size_x_m,
    pixel_size_y_m,
)


# ============================================================
# PROJECTED CRS FOR THE UNREAL LOCAL WORLD
# ============================================================
#
# Unreal itself does not provide a geographical CRS here.
#
# However, Rivelero expects a projected CRS. A LOCAL_CS is not
# considered projected by PROJ/Rasterio.
#
# We therefore embed a valid Transverse Mercator CRS whose units
# are metres.
#
# The Unreal local origin remains (0, 0).
#
# IMPORTANT:
# This CRS is SYNTHETIC.
#
# It does NOT mean that the Rural Australia environment is
# physically located at latitude 0 / longitude 0.
#
# Its purpose is to give the synthetic Unreal scene a valid
# planar projected coordinate system for GIS calculations.
# ============================================================


projected_crs = CRS.from_proj4(
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


# Verify before writing.
if not projected_crs.is_projected:
    raise RuntimeError(
        "Configured CRS is not recognised as a projected CRS."
    )


print()
print("----- CRS -----")
print(f"CRS:       {projected_crs}")
print(f"Projected: {projected_crs.is_projected}")


# ============================================================
# WRITE GEOTIFF
# ============================================================

with rasterio.open(
    OUTPUT_TIF,
    "w",
    driver="GTiff",

    width=width,
    height=height,

    count=1,
    dtype="float32",

    # This is what actually embeds the projected CRS
    # into the GeoTIFF.
    crs=projected_crs,

    transform=transform,

    nodata=NODATA_VALUE,

    compress="deflate",
) as dst:

    dst.write(
        elevation_m.astype(np.float32),
        1,
    )


    # Additional descriptive metadata.
    # These are NOT substitutes for the actual crs= parameter;
    # they simply explain how the raster was produced.
    dst.update_tags(

        source=(
            "Unreal Engine SceneCapture2D "
            "Orthographic SceneDepth DSM"
        ),

        coordinate_mode=COORDINATE_MODE,

        capture_source="SceneDepth in R",

        projection_type="Orthographic",

        projected_crs_type=(
            "Synthetic local Transverse Mercator"
        ),

        synthetic_crs="true",

        crs_wkt=projected_crs.to_wkt(),

        ortho_width_cm=str(ORTHO_WIDTH_CM),

        ortho_height_cm=str(
            ortho_height_cm
        ),

        unreal_capture_x_cm=str(
            CAPTURE_X_CM
        ),

        unreal_capture_y_cm=str(
            CAPTURE_Y_CM
        ),

        unreal_capture_z_cm=str(
            CAPTURE_Z_CM
        ),

        output_pixel_size_x_m=str(
            pixel_size_x_m
        ),

        output_pixel_size_y_m=str(
            pixel_size_y_m
        ),

        assumed_rotation=(
            "Roll=0, Pitch=270(-90), Yaw=0"
        ),

        axis_mapping=(
            "Raster columns -> Unreal Y; "
            "Raster rows -> Unreal X"
        ),

        note=(
            "Synthetic projected CRS used for Unreal "
            "local planar coordinates. "
            "Not a real-world geographic location."
        ),
    )


print()
print(f"Written: {OUTPUT_TIF}")


# ============================================================
# VERIFY OUTPUT
# ============================================================

with rasterio.open(
    OUTPUT_TIF
) as src:

    print()
    print("----- OUTPUT GEOTIFF -----")

    print(
        f"Size:         "
        f"{src.width} x {src.height}"
    )

    print(
        f"CRS:          "
        f"{src.crs}"
    )

    print(
        f"Is projected: "
        f"{src.crs.is_projected if src.crs else False}"
    )

    print(
        f"Transform:    "
        f"{src.transform}"
    )

    print(
        f"Bounds:       "
        f"{src.bounds}"
    )

    print(
        f"Resolution:   "
        f"{src.res}"
    )


    # Explicitly prove that the CRS survived into the file.
    if src.crs is None:
        raise ValueError(
            "GeoTIFF was written without a CRS."
        )


    if not src.crs.is_projected:
        raise ValueError(
            "GeoTIFF CRS is not recognised as projected.\n"
            f"CRS: {src.crs}"
        )


    out = src.read(1)


    valid_written = (
        np.isfinite(out)
        & (out != NODATA_VALUE)
    )


    if not np.any(valid_written):
        raise ValueError(
            "The written GeoTIFF contains "
            "no valid DSM pixels."
        )


    written_values = out[valid_written]


    print(
        f"DSM range:    "
        f"{written_values.min():.3f} m -> "
        f"{written_values.max():.3f} m"
    )


    print(
        f"Valid cells:  "
        f"{np.count_nonzero(valid_written):,} / "
        f"{out.size:,}"
    )


print()
print("Conversion complete.")