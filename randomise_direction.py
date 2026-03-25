#Uses a normal distribution to slightly perturb the viewing direction.
#This simulates movement of a camera as we change between images.

import numpy as np

def perturb_heading(base_heading_rad, sigma_deg=6.0, rng=None):
  
    if rng is None:
        rng = np.random.default_rng() #always create a new random number generator on each run to avoid hidden coupling.

    sigma_rad = np.deg2rad(sigma_deg) #convert any degrees to radians.
    jitter = rng.normal(loc=0.0, scale=sigma_rad) #draw a random number with our given SD and mean of 0.
    new_heading = base_heading_rad + jitter #add this number to the yaw to simulate wobble.

    return np.arctan2(np.sin(new_heading), np.cos(new_heading)) # wrap to [-pi, pi]
