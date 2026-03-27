#vista.py
from pathlib import Path
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pyproj import Transformer
import numpy as np
from matplotlib.colors import ListedColormap
from raycasting import cast_rays_360
from lineofsight import cells_crossed, line_of_sight, line_of_sight_strength, aggregate_line_of_sight
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import math


def triangle_points(E, N, spacing_m=40):

    return [
        (E, N), #the observer point
        (E + spacing_m, N), #a point ahead of the observer by spacing_m metres.
        (E + 0.5 * spacing_m, N + (math.sqrt(3) / 2) * spacing_m), #a point above the observer such that the three coordinates form an equilateral triangle.
    ]


def run_program(lon, lat, observer_height, tif_path, ax=None, show_reference=False):
    with rasterio.open(tif_path) as src:

        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True) #construct CRS transformer.
        E, N = t.transform(lon, lat) #convert from ESPG to CRS.
        if not (src.bounds.left <= E <= src.bounds.right and src.bounds.bottom <= N <= src.bounds.top):
            raise ValueError("These coordinates are outside the loaded GeoTIFF area.") #if user enters coordinates that are out of range, raise an error.

        affine = src.transform #define world pixel conversion.
        dem = src.read(1)

        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots() #create matplotlib figure and axes.
        else:
            fig = ax.figure
            ax.clear()

        show(src, ax=ax) #display DEM on axes.

        seed_points = triangle_points(E, N, spacing_m=40.0) #find coordinates of triangle.

        vis_mask = np.full(dem.shape, np.nan, dtype=np.float32) #create a vis_mask the same size as the4 DEM.

        L_region = 100 #length of region.
        h_region = L_region / 2 #half the length of region.

        for E0, N0 in seed_points:
            aggregate_line_of_sight(
                vis_mask,
                E0, N0,
                dem, src, affine,
                observer_height,
                square_size_m=L_region,
                n_rays=360,
                fade_distance=50.0
            ) #perform LoS algorithm three times, one for each coordinate of the triangle.

        left, right = E - h_region, E + h_region
        bottom, top = N - h_region, N + h_region
        #define region where plot will be zoomed in to.

        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top) #zoom in plot.

        valid = vis_mask[~np.isnan(vis_mask)] #for all the visibility values.
        if valid.size > 0:
            vmax = np.nanmax(valid) #vmax is the highest one, to scale the colours properly.
        else:
            vmax = 1.0 #or 1 by default.
        overlay = np.ma.masked_invalid(vis_mask) #convert into masked array.

        green_black_cmap = LinearSegmentedColormap.from_list(
            "green_black",
            ["black", "limegreen"]
        ) #create custom colour map from black to green (least to most visible).

        img = show(
            overlay,
            transform=affine,
            ax=ax,
            cmap=green_black_cmap,
            alpha=1.00,
            zorder=50,
            vmin=0.0,
            vmax=vmax
        )  #display mask on axes.

        overlay_im = ax.images[-1] #store the last image drawn to the axes, so that the colourbar is fitted correctly to the heatmap.

        old_cbar = getattr(fig, "_seegull_cbar", None) #search figure for preexisting colourbars.
        if old_cbar is not None:
            old_cbar.remove() #if one exists, remove it (to prevent buildup).

        fig._seegull_cbar = fig.colorbar(
            overlay_im,
            ax=ax,
            label="Visibility strength"
        ) #create and store the colourbar for visibility strength.

        ax.plot(E, N, marker='x', linestyle='None', markersize=10, markeredgewidth=2, zorder=100) #add marker to centre.

        ax.set_xlabel("Easting (m)") # axis labels
        ax.set_ylabel("Northing (m)")

        ax.set_title("Line-of-sight visibility") # axis title.

        legend_handles = [
            Line2D([0], [0], marker='x', linestyle='None', markersize=10,
                markeredgewidth=2, label="Observer")
        ] # axis legend.

        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.20, 1)) # move legend off grid to avoid overlapping.

        return {
            "overlay": overlay,
            "observer_xy": (E, N),
            "dem_path": tif_path,
            "dem_transform": affine,
            "dem_crs": src.crs,
        } #return results to GUI.py to be displayed.
