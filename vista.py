#vista.py
from pathlib import Path
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyproj import Transformer
import numpy as np
from matplotlib.colors import ListedColormap
from lineofsight import aggregate_line_of_sight
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import math
from randomisedirection import perturb_heading


def run_program(sample_metadata, tif_path, ax=None, show_reference=False):
    with rasterio.open(tif_path) as src:

        affine = src.transform #define world pixel conversion.
        dem = src.read(1)

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots() #create matplotlib figure and axes.
        else:
            fig = ax.figure
            ax.clear()

        show(src, ax=ax) #display DEM on axes.

        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) #construct CRS transformer.

        projected_samples = [] #store metadata after converting from EPSG:4326 into DEM coordinates.
        observer_points_xy = [] #store projected observer coordinates.

        for i, sample in enumerate(sample_metadata, start=1):
            lon = float(sample["lon"])
            lat = float(sample["lat"])
            observer_height = float(sample["observer_height"])
            heading_deg = float(sample.get("heading_deg", 0.0)) #default to north if heading is not supplied in the CSV.

            E, N = t.transform(lon, lat) #convert from ESPG to CRS.
            if not (src.bounds.left <= E <= src.bounds.right and src.bounds.bottom <= N <= src.bounds.top):
                raise ValueError(f"Sample {i} lies outside the loaded GeoTIFF area.") #if user enters coordinates that are out of range, raise an error.

            projected_samples.append({
                "x_coord": E,
                "y_coord": N,
                "heading_deg": heading_deg,
                "elevation_m": observer_height
            }) #for each image, its metadata is stored.

            observer_points_xy.append((E, N))

        perturbed_samples = [
                    perturb_heading(sample)
                    for sample in projected_samples
                ] #as this is a proof of concept, camera yaw will be artificially altered to simulate variation in heading between images.

        raw_overlay = np.zeros(dem.shape, dtype=np.int32) #create a mask the same size as the DEM storing raw sector footprint data on each cell.
        count_mask = np.zeros(dem.shape, dtype=np.int32) #create a mask the same size as the DEM storing visibility data on each cell.

        L_region = 100.0 #define the width of the square region inside which each sector is cast.

        for sample in perturbed_samples:
            aggregate_line_of_sight(
                raw_overlay,
                count_mask,
                sample["x_coord"],
                sample["y_coord"],
                dem,
                src,
                affine,
                observer_height=sample["elevation_m"],
                square_size_m=L_region,
                n_rays=121,
                bearing_deg=sample["heading_deg"],
                fan_angle_deg=60.0
            ) #perform LoS algorithm for each image.

        return {
            "raw_overlay": raw_overlay,
            "count_overlay": count_mask,
            "observer_points_xy": observer_points_xy,
        }