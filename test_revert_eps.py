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
    Y_c = sph_harm_y(l, abs(m), theta, phi)
    if m < 0: return np.sqrt(2) * Y_c.imag * (-1)**abs(m)
    elif m > 0: return np.sqrt(2) * Y_c.real * (-1)**abs(m)
    else: return Y_c.real

def prob_density_cart(x, y, z, n, l, m):
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-5: r = 1e-5
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    R = R_nl(r, n, l)
    Y = real_sph_harm(l, m, theta, phi)
    val = np.clip(R * Y, -1e100, 1e100)
    return val**2

def simulate(n, l, m, num_steps=100000, dt=0.01):
    x, y, z = float(n*n*a0), 0.1, 0.1
    vx, vy, vz = 0.0, 2.0, 1.0
    samples_x = np.zeros(num_steps)
    samples_y = np.zeros(num_steps)
    samples_z = np.zeros(num_steps)

    k_pot = 20.0
    d = 0.05
    max_F = 100.0

    # Old code
    eps = 1e-10

    for i in range(num_steps):
        P0 = prob_density_cart(x, y, z, n, l, m)
        Px = prob_density_cart(x+d, y, z, n, l, m)
        Py = prob_density_cart(x, y+d, z, n, l, m)
        Pz = prob_density_cart(x, y, z+d, n, l, m)

        Fx = k_pot * (Px - P0) / (d * (P0 + eps))
        Fy = k_pot * (Py - P0) / (d * (P0 + eps))
        Fz = k_pot * (Pz - P0) / (d * (P0 + eps))

        Fx = np.clip(Fx, -max_F, max_F)
        Fy = np.clip(Fy, -max_F, max_F)
        Fz = np.clip(Fz, -max_F, max_F)

        vx += Fx * dt
        vy += Fy * dt
        vz += Fz * dt

        damping = 0.99
        vx *= damping; vy *= damping; vz *= damping

        kick_strength = 0.5
        vx += np.random.normal(0, kick_strength)
        vy += np.random.normal(0, kick_strength)
        vz += np.random.normal(0, kick_strength)

        x += vx * dt; y += vy * dt; z += vz * dt
        samples_x[i] = x; samples_y[i] = y; samples_z[i] = z

    plt.figure(figsize=(6,6))
    plt.scatter(samples_x, samples_y, c='cyan', s=0.5, alpha=0.1)
    plt.title(f'Old Code: {n} {l} {m}')
    plt.savefig('test_old.png')

simulate(4, 3, 3)
simulate(1, 0, 0)
