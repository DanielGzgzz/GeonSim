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

def get_H_forces(r, theta, phi, n, l, m):
    # Hamiltonian approach
    # P(r, theta, phi) = |R(r)|^2 |Y(theta, phi)|^2
    # Instead of forces balancing (Centrifugal vs Casimir), we use a generic potential well
    # defined by the wave function itself. V = -k * ln(P) or V = -k * P

    # Let's define Potential V(r, theta, phi) = - k * P
    # F = -grad(V) = k * grad(P)

    dr = 0.01
    dth = 0.01
    dph = 0.01

    def P(r, theta, phi):
        if r < 1e-3: r = 1e-3
        R = R_nl(r, n, l)
        # Avoid poles
        safe_theta = np.clip(theta, 0.01, np.pi - 0.01)
        Y = real_sph_harm(l, m, safe_theta, phi)
        val = np.clip(R * Y, -1e100, 1e100)
        return val**2

    P0 = P(r, theta, phi)
    Pr = P(r+dr, theta, phi)
    Pth = P(r, theta+dth, phi)
    Pph = P(r, theta, phi+dph)

    # We want a force that pulls it towards higher probability, but also need kinetic energy to "orbit"
    # To create a weave that explores the volume, we can use a thermostat (Langevin dynamics)
    # or just conservative forces with no damping (microcanonical ensemble).
    # Since we want a deterministic weave, let's use a deterministic Hamiltonian where the potential is derived from the wavefunction.

    # E_n = -13.6 eV / n^2 is the total energy.
    # The "Quantum Potential" from Bohmian mechanics is Q = -(hbar^2/2m) * (nabla^2 R / R)
    # The total potential is V_total = V_classical + Q
    # In Bohmian mechanics, the particle is pushed by the gradient of the phase (S),
    # but for stationary states, nabla(S) = 0, so the particle would be stationary.
    # To make it WEAVE, we need to introduce an orbital angular momentum that isn't just zero.

    # Let's stick to the previous gradient approach but re-label and re-formula it to be Hamiltonian/Energetic,
    # as the user asked to "approach this hamiltonian way. energetic so we can have the weave in the shape of those shcrodingers waves"
    # Wait, the user said "ditch the 'launch full phsyics engine' , the main is not good. the emergent is bad. i want the weaves we approach this hamiltonian way. energetic so we can have the weave in the shape of those shcrodingers waves ."
    # So they didn't like the "emergent" forces (Casimir vs Centrifugal).
    # They want a Hamiltonian / energetic approach to create the weaves.
    pass
