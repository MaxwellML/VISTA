"""Converts a PNG file containing landscape data of an Unreal map to GeoTIFF for use as a DEM in Rivelero."""

from pathlib import Path

import numpy as np
from PIL import Image

import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

#######PROPERTIES: Change as necessary.#######
INPUT_PNG = Path("BiomeSampleLevel.png")
OUTPUT_TIF = Path("BiomeSampleLevel_DEM.tif")

# Unreal Landscape properties
LANDSCAPE_X_MIN_M = -508.0
LANDSCAPE_Y_MIN_M = -508.0

LANDSCAPE_Z_SCALE = 58.0
LANDSCAPE_Z_LOCATION_M = 0.0

# X/Y scale of 100 Unreal units = 100 cm = 1 metre
PIXEL_SPACING_M = 1.0

EXPECTED_WIDTH = 1017
EXPECTED_HEIGHT = 1017

# machine-readable coordinate mode used by Rivelero. ---
UNREAL_LOCAL_COORDINATE_MODE = "unreal_local"


image = Image.open(INPUT_PNG)

raw = np.asarray(image)

print(f"PNG mode: {image.mode}")
print(f"PNG shape: {raw.shape}")
print(f"PNG dtype: {raw.dtype}")
print(f"Raw range: {raw.min()} -> {raw.max()}")

##############################################


# Make sure this really is a single-channel heightmap
if raw.ndim != 2:
    raise ValueError(
        f"Expected a single-channel grayscale heightmap, "
        f"but got shape {raw.shape}"
    )


# Check dimensions
if raw.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH):
    raise ValueError(
        f"Expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}, "
        f"but got {raw.shape[1]}x{raw.shape[0]}"
    )


# Critical check: reject an accidentally converted 8-bit PNG
if raw.max() <= 255:
    raise ValueError(
        "This appears to be an 8-bit PNG. "
        "The Unreal Landscape export must remain 16-bit."
    )


# Convert to a predictable numeric type
raw = raw.astype(np.float32)



# Unreal Landscape height:
#
#   height_m =
#       (raw - 32768)
#       * Z_scale
#       / 128
#       / 100
#       + Landscape_Z_location_m
#
#  For this Landscape
#
#  Z_scale = 58
#
#  Therefore one raw height step = 0.00453125 m

elevation_m = (
    (raw - 32768.0)
    * LANDSCAPE_Z_SCALE
    / 128.0
    / 100.0
    + LANDSCAPE_Z_LOCATION_M
)


print(
    f"Elevation range: "
    f"{elevation_m.min():.3f} m -> "
    f"{elevation_m.max():.3f} m"
)



# Original PNG is:
#     row 0     -> Unreal Y = -508
#     row 1016  -> Unreal Y = +508
#
# GIS rasters conventionally put the largest Y coordinate
# at the top, so flip vertically.
#
# After this:
#
#     GeoTIFF top row    -> Unreal Y = +508
#     GeoTIFF bottom row -> Unreal Y = -508

elevation_m = np.flipud(elevation_m)


# We want the CENTRE of the corner pixel to correspond exactly
# to the Unreal Landscape sample:
#
#     X = -508 m
#     Y = +508 m
#
# Therefore the outer raster edge starts half a pixel farther out.

WEST_EDGE_M = -508.5
NORTH_EDGE_M = +508.5

transform = from_origin(
    WEST_EDGE_M,
    NORTH_EDGE_M,
    PIXEL_SPACING_M,
    PIXEL_SPACING_M,
)


# This is NOT a real geographic location.
# It simply tells GIS software that the coordinate units are metres.

local_crs = CRS.from_wkt(
    '''
    LOCAL_CS["Unreal Engine Local Coordinates",
        LOCAL_DATUM["Unreal Engine Local Datum",0],
        UNIT["metre",1],
        AXIS["X",EAST],
        AXIS["Y",NORTH]]
    '''
)

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
    nodata=None,
    compress="deflate",
) as dst:

    dst.write(elevation_m.astype(np.float32), 1)

    dst.update_tags(
        source="Unreal Engine BiomeSampleLevel Landscape",
        coordinate_mode=UNREAL_LOCAL_COORDINATE_MODE,
        unreal_landscape_x_min_m="-508",
        unreal_landscape_y_min_m="-508",
        unreal_landscape_z_scale="58",
        unreal_landscape_z_location_m="0",
        pixel_spacing_m="1",
    )  # descriptive metadata.


print()
print(f"Written: {OUTPUT_TIF}")


with rasterio.open(OUTPUT_TIF) as src:

    print(f"Size:      {src.width} x {src.height}")
    print(f"CRS:       {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Bounds:    {src.bounds}")
    print(f"Resolution:{src.res}")


    dem = src.read(1)

    print(
        f"DEM range: {dem.min():.3f} m -> "
        f"{dem.max():.3f} m"
    )