import numpy as np
from scipy.special import sph_harm_y
import math

def real_sph_harm_old(l, m, theta, phi):
    Y_c = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * (-1)**m * Y_c.imag
    elif m > 0:
        return np.sqrt(2) * (-1)**m * Y_c.real
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
        return math.sqrt(2) * norm * P * math.cos(m * phi)
    elif m < 0:
        return math.sqrt(2) * norm * P * math.sin(m_abs * phi)
    else:
        return norm * P

# Real spherical harmonics mathematically standard definition (including condon-shortley)
# Y_l^m_real =
# if m > 0: 1/sqrt(2) * (Y_l^m + (-1)^m Y_l^{-m})
# if m < 0: 1/(sqrt(2)i) * (Y_l^{-m} - (-1)^m Y_l^m)

# Since Y_l^{-m} = (-1)^m (Y_l^m)*
# If m > 0:
# 1/sqrt(2) (Y_l^m + (-1)^m (-1)^m (Y_l^m)*) = 1/sqrt(2) (Y_l^m + (Y_l^m)*) = sqrt(2) * Re(Y_l^m)
# If m < 0: let m_abs = |m|. We want Y_l^{-|m|} = (-1)^|m| (Y_l^|m|)*
# No, standard real spherical harmonics for m < 0:
# Y_l^m_real = 1/(sqrt(2)i) (Y_l^|m| - (-1)^|m| Y_l^{-|m|})
# = 1/(sqrt(2)i) (Y_l^|m| - (-1)^|m| (-1)^|m| (Y_l^|m|)*) = 1/(sqrt(2)i) (Y_l^|m| - (Y_l^|m|)*) = sqrt(2) * Im(Y_l^|m|)
# But scipy sph_harm_y has condon-shortley built in.
# Let's test with sph_harm_y(l, m, theta, phi)

def real_sph_harm_correct(l, m, theta, phi):
    Y_c = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * Y_c.imag * (-1)**abs(m)
    elif m > 0:
        return np.sqrt(2) * Y_c.real * (-1)**abs(m)
    else:
        return Y_c.real

theta, phi = 1.2, 0.5
for l in range(4):
    for m in range(-l, l+1):
        o = real_sph_harm_old(l, m, theta, phi)
        j = real_sph_harm_js(l, m, theta, phi)
        c = real_sph_harm_correct(l, m, theta, phi)
        print(f"l={l}, m={m}: old={o:.4f}, js={j:.4f}, correct={c:.4f}")
