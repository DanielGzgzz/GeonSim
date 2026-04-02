import numpy as np
from scipy.special import sph_harm_y
import math

def real_sph_harm_correct(l, m, theta, phi):
    Y_c = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * Y_c.imag * (-1)**abs(m)
    elif m > 0:
        return np.sqrt(2) * Y_c.real * (-1)**abs(m)
    else:
        return Y_c.real

def prob_density_cart(x, y, z, n, l, m):
    """Probability density in Cartesian coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-5:
        r = 1e-5
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    Y = real_sph_harm_correct(l, m, theta, phi)
    return Y

# let's look at 4f, which means n=4, l=3
# x, y, z on a sphere
pts = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 1),
]

for p in pts:
    norm = np.linalg.norm(p)
    x, y, z = [v/norm for v in p]
    print(f"pt {p}: Y(4f, m=3) = {prob_density_cart(x, y, z, 4, 3, 3)}")
