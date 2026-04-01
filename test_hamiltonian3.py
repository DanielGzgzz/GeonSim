import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y, genlaguerre
import math
import os

a0 = 1.0  # Bohr radius

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
    val = np.clip(R * Y, -1e100, 1e100)
    return val**2

def simulate_hamiltonian_orbit(n, l, m, num_steps=100000, dt=0.02):
    """
    Simulates a trajectory using a Hamiltonian energetic approach.
    The particle is drawn towards high-probability regions by a Quantum Potential-like
    gradient, but maintains orbital momentum.
    """
    # 1. Initial State
    # Start near the expected radius
    r = n**2 * a0 * 0.9
    theta = np.pi / 4.0 if l > 0 else np.pi / 2.0
    phi = 0.0

    # Give it some initial kinetic energy (momentum) to orbit
    vr = 0.0
    vtheta = 0.2
    vphi = 0.2

    samples_x = np.zeros(num_steps)
    samples_y = np.zeros(num_steps)
    samples_z = np.zeros(num_steps)

    dr = 0.01
    d_theta = 0.01
    d_phi = 0.01

    # Scale factor for the artificial "potential" derived from the wavefunction
    k_pot = 0.5

    for i in range(num_steps):
        # We treat the probability density as an inverted potential well: V(r,th,phi) = -k * P(r,th,phi)
        # Force is F = -grad(V) = k * grad(P)

        safe_theta = np.clip(theta, 0.01, np.pi - 0.01)
        safe_r = max(r, 0.01)

        # Current Density
        p_current = probability_density(safe_r, safe_theta, phi, n, l, m)

        # Gradients
        p_r_plus = probability_density(safe_r + dr, safe_theta, phi, n, l, m)
        p_theta_plus = probability_density(safe_r, safe_theta + d_theta, phi, n, l, m)
        p_phi_plus = probability_density(safe_r, safe_theta, phi + d_phi, n, l, m)

        grad_r = (p_r_plus - p_current) / dr
        grad_theta = (p_theta_plus - p_current) / d_theta
        grad_phi = (p_phi_plus - p_current) / d_phi

        # Normalize force by current density to create a logarithmic well (like Quantum Potential)
        # F = grad(P) / P  is proportional to grad(ln P)
        norm_factor = (p_current + 1e-10)

        F_r = (grad_r / norm_factor) * k_pot
        F_theta = (grad_theta / norm_factor) * k_pot
        F_phi = (grad_phi / norm_factor) * k_pot

        # Add angular momentum to prevent collapsing into the center
        # For non-zero l, there's a centrifugal barrier: F_centrifugal = l*(l+1) / r^3
        # We can explicitly add this centrifugal force to keep it out of the origin
        if l > 0:
            F_r += (l * (l + 1) * 0.5) / (safe_r**3)

        # Add constant phi rotation to sweep the volume if m is non-zero
        F_phi += (0.01 * (abs(m) + 1))

        # Clamp to prevent instability
        F_r = np.clip(F_r, -2.0, 2.0)
        F_theta = np.clip(F_theta, -0.5, 0.5)
        F_phi = np.clip(F_phi, -0.5, 0.5)

        # Euler Integration
        vr += F_r * dt
        vtheta += F_theta * dt
        vphi += F_phi * dt

        # Small damping to represent "coupling" to the energetic field (thermostat)
        # Without damping, numerical errors cause the orbit to fly apart.
        vr *= 0.99
        vtheta *= 0.99
        vphi *= 0.999

        r += vr * dt
        theta += vtheta * dt
        phi += vphi * dt

        # Store
        samples_x[i] = r * np.sin(theta) * np.cos(phi)
        samples_y[i] = r * np.sin(theta) * np.sin(phi)
        samples_z[i] = r * np.cos(theta)

    r_out = np.sqrt(samples_x**2 + samples_y**2 + samples_z**2)
    theta_out = np.arccos(samples_z / (r_out + 1e-10))
    phi_out = np.arctan2(samples_y, samples_x)

    return r_out, theta_out, phi_out

xs, ys, zs = simulate_hamiltonian_orbit(4, 3, 1)
