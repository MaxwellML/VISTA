#raycasting.py
#find direction vector for a given angle.
#cast ray along vector until bounding box edge hit.
#repeat incrementally across 2pi radians.

import math
from typing import Optional, Tuple, List
from rasterio.transform import rowcol


def ray_hit_square(
    E0: float,
    N0: float,
    half_size_m: float,
    theta_rad: float,
    eps: float = 1e-12,
):

    eps = 1e-12
    dE = math.cos(theta_rad)
    dN = math.sin(theta_rad) #create unit vector from N and E components.

    L = half_size_m * 2
    half = L / 2
    Emin, Emax = E0 - half, E0 + half
    Nmin, Nmax = N0 - half, N0 + half #define region of interest.

    candidates: List[Tuple[float, float, float]] = []  #define list for candidate intersection points. (prevents overshotting)

    if abs(dE) > eps: #if we are on course for a vertical wall.
        for E_side in (Emin, Emax): #for either side of the square.
            t = (E_side - E0) / dE #find distance needed to travel to hit wall.
            if t > 0: #a negative value indicates we'd travel backwards to reach the wall, not valid.
                N = N0 + t * dN
                if Nmin - 1e-9 <= N <= Nmax + 1e-9:
                    candidates.append((t, E_side, N)) #store potential wall hit.

    if abs(dN) > eps: #if we are on course for a horizontal wall.
        for N_side in (Nmin, Nmax): #for either side of the square.
            t = (N_side - N0) / dN  #find distance needed to travel to hit wall.
            if t > 0: #a negative value indicates we'd travel backwards to reach the wall, not valid.
                E = E0 + t * dE
                if Emin - 1e-9 <= E <= Emax + 1e-9:
                    candidates.append((t, E, N_side)) #store potential wall hit.

    if not candidates:
        return None

    t_hit, E_hit, N_hit = min(candidates, key=lambda x: x[0]) #minimum t corresponds to actual wall hit.

    return (E_hit, N_hit)


def cast_rays_360(
    E0: float,
    N0: float,
    square_size_m: float = 100.0,
    n_rays: int = 360,
    affine=None,
    heading_deg: Optional[float] = None, 
    fan_angle_deg: float = 360.0
    
):

    if affine is None: #if no affine passed, raise an error.
        raise ValueError("affine must be provided")

    r, c = rowcol(affine, E0, N0) #define row and column of centre.
    half = square_size_m / 2
    hits: List[Tuple[float, float]] = []

    centre_theta = math.radians(90 - heading_deg) #since 0 is east in Python convention, the angle must be converted to a compass bearing.
    half_fan = math.radians(fan_angle_deg) / 2.0 #sector is centre +- half_fan to ensure symmetry.

    thetas = [
                centre_theta - half_fan + k * (2 * half_fan / (n_rays - 1))
                for k in range(n_rays)
    ] #angles to be swept through.

    
    for theta in thetas:
        hit = ray_hit_square(E0, N0, half, theta) #find ray for a given angle.
        if hit is None:
            continue
        Eh, Nh = hit #find point of intersection.
        hits.append((Eh, Nh))

    return hits
