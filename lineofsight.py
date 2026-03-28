#lineofsight.py
#once ray is constructed, find the cells that it intersects.
#compute slope to each cell.
#cell is visible iff it is not behind a cell with a greater slope.
#repeat for each coordinate, then aggregate.

import numpy as np
import math
from typing import Iterable, Tuple, Optional
from affine import Affine
from rasterio.transform import rowcol
from raycasting import cast_rays_360


def cell_centre(affine: Affine, r: int, c: int): #compute centre of cell.
    E, N = affine * (c + 0.5, r + 0.5)
    return E, N


def aggregate_line_of_sight(raw_mask, count_mask, E0, N0, dem, src, affine, observer_height,
                        square_size_m=100, n_rays=360, bearing_deg=None, fan_angle_deg=360.0):

        hits = cast_rays_360(E0, N0, square_size_m=square_size_m, n_rays=n_rays, affine=affine, bearing_deg=bearing_deg,
        fan_angle_deg=fan_angle_deg) #cast rays.
        raw_this_view = np.zeros_like(raw_mask, dtype=bool) #a mask with exactly the same function as raw_mask, just temporary.
        seen_this_view = np.zeros_like(count_mask, dtype=bool) #a mask with exactly the same function as count_mask, just temporary.
        for (Eh, Nh) in hits: #for the length of each ray.
            cells = list(cells_crossed(affine, src.width, src.height, E0, N0, Eh, Nh)) #compute which cells were passed through.
            if not cells: #in case none were found, skip this ray.
                continue

            for (r, c) in cells:
                raw_this_view[r, c] = True #if cell lies anywhere inside the swept sector, mark it in the raw overlay.

            visible = line_of_sight(
            cells,
            dem,
            affine,
            E0,
            N0,
            observer_height,
            nodata=src.nodata
        ) #retrieve visibility data on each cell, either seen or not seen.

            for (r, c), is_visible in zip(cells, visible):
                if is_visible:
                    seen_this_view[r, c] = True #if cell is seen at all, mark it as visible.

        raw_mask[raw_this_view] += 1 #increment 1 to the cells that lay anywhere inside the sector footprint.
        count_mask[seen_this_view] += 1 #increment 1 to the cells that were seen at all (note: only 1 is added to avoid overcounting from distinct rays intersecting the same cell).
        return raw_mask, count_mask

def line_of_sight(
    cells: Iterable[Tuple[int, int]],
    dem,                      # 2D array: dem[r][c] or dem[r, c]
    affine: Affine,
    E0: float,
    N0: float,
    observer_height: float = 0,
    nodata: Optional[float] = None,
    eps: float = 1e-12,
):

    cells = list(cells) #convert iterable to list to allow indexing.
    if not cells:
        return []

    r0, c0 = cells[0] #begin at first cell (observer's point).
    z0 = float(dem[r0, c0]) + observer_height #find DEM elevation at observer's height.

    visible = [True]
    max_slope = -math.inf #track the maximum slope seen so far (any slope is greater than negative infinity, so default to that).

    for (r, c) in cells[1:]:
        z = float(dem[r, c]) #read terrain height of cell.
        if nodata is not None and z == nodata: #if there is an issue, mark as "invisible" by default.
            visible.append(False)
            continue

        E, N = cell_centre(affine, r, c) #find world coordinate of cell's centre.

        d = math.hypot(E - E0, N - N0) #find distance from observer to cell.
        if d < eps:
            visible.append(True)
            continue

        s = (z - z0) / d #find slope from observer to cell.

        if s > max_slope: #if slope is greater than the previous max slope, update max slope.
            visible.append(True) #make it visible.
            max_slope = s
        else:
            visible.append(False) #else it cannot be visible since it lies behind the existing max slope.

    return visible #return the visibility list for each cell.



def cells_crossed(
    affine: Affine,
    width: int,
    height: int,
    E0: float,
    N0: float,
    E1: float,
    N1: float,
    eps: float = 1e-12,
):

    dE = E1 - E0
    dN = N1 - N0 #define direction vector given start and end points.

    r, c = rowcol(affine, E0, N0) #define r,c of start point
    r_end, c_end = rowcol(affine, E1, N1) #define r,c of end point

    if not (0 <= r < height and 0 <= c < width):
        return #do not allow indexing outside the raster grid.

    yield (r, c) #output starting cell, then continue.

    resE = affine.a
    resN = -affine.e  #find the width and height of a single pixel as defined by the raster file.

    step_c = 1 if dE > 0 else (-1 if dE < 0 else 0) #find if ray points east or west.
    step_r = 1 if dN < 0 else (-1 if dN > 0 else 0) #find if ray points north or south.

    x_left, y_top = affine * (c, r)
    x_right = x_left + resE
    y_bottom = y_top - resN #find rectangle bounds of current cell as world coordinates.

    def safe_div(num: float, den: float) -> float:
        return num / den if abs(den) > eps else math.inf #avoid division by 0 if we are perfectly horizontal or vertical.

    if step_c > 0:
        tMaxX = safe_div(x_right - E0, dE)
    elif step_c < 0:
        tMaxX = safe_div(x_left - E0, dE)
    else:
        tMaxX = math.inf

    #find how far we would need to travel to hit a horizontal border.

    if step_r > 0:
        tMaxY = safe_div(y_bottom - N0, dN)
    elif step_r < 0:
        tMaxY = safe_div(y_top - N0, dN)
    else:
        tMaxY = math.inf

    #find how far we would need to travel to hit a vertical border.

    tDeltaX = abs(resE / dE) if abs(dE) > eps else math.inf #set jump size for vertical boundary lines.
    tDeltaY = abs(resN / dN) if abs(dN) > eps else math.inf #set jump size for horiztontal boundary lines.

    while (r, c) != (r_end, c_end): #until we hit the boundary line.
        if tMaxX + eps < tMaxY: #if we will hit a vertical boundary first, move into neighbouring column.
            c += step_c
            tMaxX += tDeltaX
        elif tMaxY + eps < tMaxX: #if we will hit a vertical boundary first, move into neighbouring row.
            r += step_r
            tMaxY += tDeltaY
        else: #if we will hit a corner, move into diagonally neighbouring square.
            c += step_c
            r += step_r
            tMaxX += tDeltaX
            tMaxY += tDeltaY

        if not (0 <= r < height and 0 <= c < width):
            return

        yield (r, c) #output new cell we have moved into.
