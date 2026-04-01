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
    val = np.clip(R * Y, -1e100, 1e100)
    return val**2

def simulate_hamiltonian_orbit(n, l, m, num_steps=100000, dt=0.05):
    # Base energy state
    r0 = n**2 * a0
    # Potential scaling
    k_e = 1.0 # Coulomb
    e2 = 1.0

    r = r0 * 0.9
    theta = np.pi / 4.0 if l > 0 else np.pi / 2.0
    phi = 0.0

    vr, vtheta, vphi = 0.0, 0.1, 0.1

    samples_x = np.zeros(num_steps)
    samples_y = np.zeros(num_steps)
    samples_z = np.zeros(num_steps)

    d_theta = 0.01
    d_phi = 0.01

    for i in range(num_steps):
        # We replace the Emergent Casimir vs Centrifugal force with
        # an Energetic / Hamiltonian approach based on Quantum Potential (Q)
        # where the particle is steered by the gradient of the probability amplitude.
        # F = -grad(V + Q). We approximate this by steering the particle down the
        # gradients of the probability density field, simulating a "Quantum Potential well"
        # that binds the particle to the shape of the Schrödinger wave.

        # 1. Classical Coulomb Potential (Inward) vs Kinetic Energy (Outward)
        # We simplify the radial binding to a central potential well
        # F_classical = -k_e * e^2 / r^2 + (angular momentum barrier)
        # However, to maintain the orbital shape without complex scattering,
        # we can use the radial wavefunction gradient as the radial quantum force.

        safe_theta = np.clip(theta, 0.01, np.pi - 0.01)
        safe_r = max(r, 0.01)

        # Calculate numerical gradients of the Quantum Potential field (probability density)
        # We use P = |psi|^2. The force is F_Q = k * grad(P) / P
        p_current = probability_density(safe_r, safe_theta, phi, n, l, m)
        p_theta_plus = probability_density(safe_r, safe_theta + d_theta, phi, n, l, m)
        p_phi_plus = probability_density(safe_r, safe_theta, phi + d_phi, n, l, m)

        # We also need radial gradient for full Hamiltonian
        dr = 0.01
        p_r_plus = probability_density(safe_r + dr, safe_theta, phi, n, l, m)

        grad_r = (p_r_plus - p_current) / dr
        grad_theta = (p_theta_plus - p_current) / d_theta
        grad_phi = (p_phi_plus - p_current) / d_phi

        # Normalize the steering force
        steer_strength = 0.05
        norm_factor = (p_current + 1e-10)

        if np.isnan(grad_r) or np.isinf(grad_r): grad_r = 0.0
        if np.isnan(grad_theta) or np.isinf(grad_theta): grad_theta = 0.0
        if np.isnan(grad_phi) or np.isinf(grad_phi): grad_phi = 0.0

        # Quantum Forces derived from the Potential Q
        F_r = (grad_r / norm_factor) * steer_strength
        F_theta = (grad_theta / norm_factor) * steer_strength
        F_phi = (grad_phi / norm_factor) * steer_strength + (0.01 * (abs(m) + 1))

        # To make it a stable weave, we combine the classical Coulomb force
        # with the centrifugal barrier, and use the quantum force to steer the angles.
        # F_classical_r = (E_n / r) - (K_vac / r**2)
        # But wait, the user explicitly asked to "ditch the emergent... approach this hamiltonian way. energetic"
        # Let's completely use the gradients of the wavefunction for ALL forces:
        # F = k * grad(P) / P

        # Clamp forces
        if np.isnan(F_r) or np.isinf(F_r): F_r = 0.0
        if np.isnan(F_theta) or np.isinf(F_theta): F_theta = 0.0
        if np.isnan(F_phi) or np.isinf(F_phi): F_phi = 0.0
        F_r = np.clip(F_r, -0.5, 0.5)
        F_theta = np.clip(F_theta, -0.5, 0.5)
        F_phi = np.clip(F_phi, -0.5, 0.5)

        # Euler Integration (Update Kinetic Energy/Momenta)
        vr += F_r * dt
        vtheta += F_theta * dt
        vphi += F_phi * dt

        # Hamiltonian "Thermostat" damping to keep the particle bound within the energetic well
        vr *= 0.98
        vtheta *= 0.99
        vphi *= 0.999

        # Update Position
        r += vr * dt
        theta += vtheta * dt
        phi += vphi * dt

        samples_x[i] = r * np.sin(theta) * np.cos(phi)
        samples_y[i] = r * np.sin(theta) * np.sin(phi)
        samples_z[i] = r * np.cos(theta)

    r_out = np.sqrt(samples_x**2 + samples_y**2 + samples_z**2)
    theta_out = np.arccos(samples_z / (r_out + 1e-10))
    phi_out = np.arctan2(samples_y, samples_x)

    return r_out, theta_out, phi_out

xs, ys, zs = simulate_hamiltonian_orbit(4, 3, 1)
x = xs * np.sin(ys) * np.cos(zs)
y = xs * np.sin(ys) * np.sin(zs)
z = xs * np.cos(ys)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=0.1, alpha=0.1)
plt.savefig('test_ham2.png')
