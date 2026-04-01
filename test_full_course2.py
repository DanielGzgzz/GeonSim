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

def simulate_cartesian(n, l, m, num_steps=200000, dt=0.01):
    x, y, z = float(n*n*a0), 0.1, 0.1
    vx, vy, vz = 0.0, 2.0, 1.0

    xs, ys, zs = np.zeros(num_steps), np.zeros(num_steps), np.zeros(num_steps)
    k = 20.0 # strong potential well
    d = 0.05

    for i in range(num_steps):
        P0 = prob_density_cart(x, y, z, n, l, m)
        Px = prob_density_cart(x+d, y, z, n, l, m)
        Py = prob_density_cart(x, y+d, z, n, l, m)
        Pz = prob_density_cart(x, y, z+d, n, l, m)

        # F = k * grad(P) / P
        eps = 1e-10
        Fx = k * (Px - P0) / (d * (P0 + eps))
        Fy = k * (Py - P0) / (d * (P0 + eps))
        Fz = k * (Pz - P0) / (d * (P0 + eps))

        # Clamp forces
        max_F = 100.0
        Fx = np.clip(Fx, -max_F, max_F)
        Fy = np.clip(Fy, -max_F, max_F)
        Fz = np.clip(Fz, -max_F, max_F)

        vx += Fx * dt
        vy += Fy * dt
        vz += Fz * dt

        # Damping to act as a thermostat and keep it inside the lobes
        damping = 0.99
        vx *= damping
        vy *= damping
        vz *= damping

        # Add a stochastic kick (Langevin dynamics) to make it explore the full volume
        # instead of settling into a single ring trajectory
        kick = 0.5
        vx += np.random.normal(0, kick)
        vy += np.random.normal(0, kick)
        vz += np.random.normal(0, kick)

        x += vx * dt
        y += vy * dt
        z += vz * dt

        xs[i] = x
        ys[i] = y
        zs[i] = z

    return xs, ys, zs

x, y, z = simulate_cartesian(4, 3, 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=0.1, alpha=0.1)
plt.savefig('test_full2.png')
