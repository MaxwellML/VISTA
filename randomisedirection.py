#Uses a normal distribution to slightly perturb the viewing direction.
#This simulates movement of a camera as we change between images.

import numpy as np

def perturb_heading(sample_metadata, sigma_deg=6.0, rng=None):
  
    if rng is None:
        rng = np.random.default_rng() #always create a new random number generator on each run to avoid hidden coupling.

    perturbed_sample = sample_metadata.copy() #create a separate copy so the original metadata is not overwritten.
    base_heading_deg = perturbed_sample["heading_deg"] #retrieve heading.
    jitter = rng.normal(loc=0.0, scale=sigma_deg) #draw a random number with our given SD and mean of 0.

    perturbed_sample["heading_deg"] = (base_heading_deg + jitter) % 360 #wrap heading back into the range [0, 360).

    return perturbed_sample