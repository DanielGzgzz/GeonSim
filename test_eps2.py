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
    val = np.clip(R_nl(r, n, l) * real_sph_harm(l, m, theta, phi), -1e100, 1e100)
    return val**2

def sim(eps_val):
    n, l, m = 3, 2, 2
    x, y, z = float(n*n*a0), 0.1, 0.1
    vx, vy, vz = 0.0, 2.0, 1.0
    num_steps = 100000
    dt = 0.01
    k_pot = 20.0
    d = 0.05
    sx, sy, sz = [], [], []

    for _ in range(num_steps):
        P0 = prob_density_cart(x, y, z, n, l, m)
        Px = prob_density_cart(x+d, y, z, n, l, m)
        Py = prob_density_cart(x, y+d, z, n, l, m)
        Pz = prob_density_cart(x, y, z+d, n, l, m)

        Fx = k_pot * (Px - P0) / (d * (P0 + eps_val))
        Fy = k_pot * (Py - P0) / (d * (P0 + eps_val))
        Fz = k_pot * (Pz - P0) / (d * (P0 + eps_val))

        max_F = 100.0
        Fx = np.clip(Fx, -max_F, max_F)
        Fy = np.clip(Fy, -max_F, max_F)
        Fz = np.clip(Fz, -max_F, max_F)

        vx += Fx * dt
        vy += Fy * dt
        vz += Fz * dt

        damping = 0.99
        vx *= damping
        vy *= damping
        vz *= damping

        kick_strength = 0.5
        vx += np.random.normal(0, kick_strength)
        vy += np.random.normal(0, kick_strength)
        vz += np.random.normal(0, kick_strength)

        x += vx * dt
        y += vy * dt
        z += vz * dt

        sx.append(x)
        sy.append(y)
        sz.append(z)

    return sx, sy, sz

np.random.seed(42)
sx1, sy1, sz1 = sim(1e-10)
np.random.seed(42)
sx2, sy2, sz2 = sim(1e-8)
np.random.seed(42)
sx3, sy3, sz3 = sim(1e-6)
np.random.seed(42)
sx4, sy4, sz4 = sim(1e-4)

fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(sx1, sy1, sz1, c='cyan', s=0.5, alpha=0.1, edgecolors='none')
ax1.set_title("eps=1e-10")

ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(sx2, sy2, sz2, c='cyan', s=0.5, alpha=0.1, edgecolors='none')
ax2.set_title("eps=1e-8")

ax3 = fig.add_subplot(143, projection='3d')
ax3.scatter(sx3, sy3, sz3, c='cyan', s=0.5, alpha=0.1, edgecolors='none')
ax3.set_title("eps=1e-6")

ax4 = fig.add_subplot(144, projection='3d')
ax4.scatter(sx4, sy4, sz4, c='cyan', s=0.5, alpha=0.1, edgecolors='none')
ax4.set_title("eps=1e-4")

plt.savefig('test_eps2.png', dpi=150)
print("done")
