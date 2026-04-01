import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y, genlaguerre
import math

a0 = 1.0

def R_nl(r, n, l):
    rho = 2.0 * r / (n * a0)
    norm = np.sqrt((2.0 / (n * a0))**3 * math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l)))
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * np.exp(-rho / 2.0) * rho**l * laguerre

def real_sph_harm(l, m, theta, phi):
    Y_c = sph_harm_y(l, abs(m), phi, theta)
    if m < 0:
        return np.sqrt(2) * (-1)**m * Y_c.imag
    elif m > 0:
        return np.sqrt(2) * (-1)**m * Y_c.real
    else:
        return Y_c.real

def prob_density_cart(x, y, z, n, l, m):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-5:
        r = 1e-5
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    R = R_nl(r, n, l)
    Y = real_sph_harm(l, m, theta, phi)
    return (R * Y)**2

def simulate(n, l, m, steps=50000, dt=0.01):
    x, y, z = float(n*n*a0), 0.1, 0.1 # start slightly off-axis
    vx, vy, vz = 0.0, 1.0, 0.5 # initial kinetic energy

    xs, ys, zs = [], [], []
    k = 0.5 # Hamiltonian potential scale
    eps = 1e-10
    d = 0.001

    for i in range(steps):
        P0 = prob_density_cart(x, y, z, n, l, m)
        Px = prob_density_cart(x+d, y, z, n, l, m)
        Py = prob_density_cart(x, y+d, z, n, l, m)
        Pz = prob_density_cart(x, y, z+d, n, l, m)

        # F = k * grad(P) / P  (which is grad(k*ln(P)))
        Fx = k * (Px - P0) / (d * (P0 + eps))
        Fy = k * (Py - P0) / (d * (P0 + eps))
        Fz = k * (Pz - P0) / (d * (P0 + eps))

        # Clamp forces
        max_F = 50.0
        Fx = np.clip(Fx, -max_F, max_F)
        Fy = np.clip(Fy, -max_F, max_F)
        Fz = np.clip(Fz, -max_F, max_F)

        vx += Fx * dt
        vy += Fy * dt
        vz += Fz * dt

        # Slight damping to settle into the well
        vx *= 0.999
        vy *= 0.999
        vz *= 0.999

        # Optional: Add small transverse kick if velocity is too low to maintain weave?
        # A true Hamiltonian might need a thermostat or just zero damping to weave ergodically.

        x += vx * dt
        y += vy * dt
        z += vz * dt

        xs.append(x)
        ys.append(y)
        zs.append(z)

    return xs, ys, zs

xs, ys, zs = simulate(4, 3, 1) # 4f
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, s=0.1, alpha=0.1)
plt.savefig('test_ham.png')
