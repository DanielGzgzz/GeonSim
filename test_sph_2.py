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

def legendreP(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = math.sqrt((1.0 - x)*(1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm

    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1

    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll

    return pll

def real_sph_harm_js(l, m, theta, phi):
    m_abs = abs(m)
    norm = math.sqrt(((2*l + 1)*math.factorial(l - m_abs)) / (4*math.pi*math.factorial(l + m_abs)))
    P = legendreP(l, m_abs, math.cos(theta))

    if m > 0:
        return math.sqrt(2) * norm * P * math.cos(m * phi) * (-1)**m_abs
    elif m < 0:
        return math.sqrt(2) * norm * P * math.sin(m_abs * phi) * (-1)**m_abs
    else:
        return norm * P

theta, phi = 1.2, 0.5
for l in range(4):
    for m in range(-l, l+1):
        j = real_sph_harm_js(l, m, theta, phi)
        c = real_sph_harm_correct(l, m, theta, phi)
        if abs(j - c) > 1e-5:
            print(f"Mismatch l={l}, m={m}: js={j:.4f}, correct={c:.4f}")
