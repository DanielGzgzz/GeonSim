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

def probability_density(r, theta, phi, n, l, m):
    R = R_nl(r, n, l)
    Y = real_sph_harm(l, m, theta, phi)
    val = np.clip(R * Y, -1e10, 1e10)
    return val**2

def simulate(n, l, m, num_steps=200000, dt=0.01):
    r0 = n**2 * a0
    r = r0 * 0.5
    theta = np.pi / 4.0 if l > 0 else np.pi / 2.0
    phi = 0.0

    # High initial kinetic energy to swing through the well
    vr, vtheta, vphi = 1.0, 2.0, 1.5

    samples_x = np.zeros(num_steps)
    samples_y = np.zeros(num_steps)
    samples_z = np.zeros(num_steps)

    dr = 0.05
    d_theta = 0.05
    d_phi = 0.05

    for i in range(num_steps):
        safe_theta = np.clip(theta, 0.01, np.pi - 0.01)
        safe_r = max(r, 0.01)

        p_current = probability_density(safe_r, safe_theta, phi, n, l, m)
        p_r_plus = probability_density(safe_r + dr, safe_theta, phi, n, l, m)
        p_theta_plus = probability_density(safe_r, safe_theta + d_theta, phi, n, l, m)
        p_phi_plus = probability_density(safe_r, safe_theta, phi + d_phi, n, l, m)

        grad_r = (p_r_plus - p_current) / dr
        grad_theta = (p_theta_plus - p_current) / d_theta
        grad_phi = (p_phi_plus - p_current) / d_phi

        norm_factor = (p_current + 1e-10)

        # V = -k * ln(P) -> F = k * grad(P) / P
        steer = 10.0 # Increase force to make it snap back
        F_r = (grad_r / norm_factor) * steer
        F_theta = (grad_theta / norm_factor) * steer
        F_phi = (grad_phi / norm_factor) * steer

        # Add a centrifugal barrier to avoid r=0 singularity explosion
        if safe_r < 0.1:
             F_r += 1.0 / (safe_r**3)

        # Ensure no explosive NaN
        F_r = np.clip(F_r, -50.0, 50.0)
        F_theta = np.clip(F_theta, -50.0, 50.0)
        F_phi = np.clip(F_phi, -50.0, 50.0)

        vr += F_r * dt
        vtheta += F_theta * dt
        vphi += F_phi * dt

        # VERY light damping just to keep it chaotic but bounded
        vr *= 0.999
        vtheta *= 0.999
        vphi *= 0.999

        r += vr * dt
        theta += vtheta * dt
        phi += vphi * dt

        samples_x[i] = r * np.sin(theta) * np.cos(phi)
        samples_y[i] = r * np.sin(theta) * np.sin(phi)
        samples_z[i] = r * np.cos(theta)

    return samples_x, samples_y, samples_z

x, y, z = simulate(4, 3, 1)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=0.1, alpha=0.1)
plt.savefig('test_full.png')
