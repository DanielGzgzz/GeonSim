import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y, genlaguerre
import math
import os

# Create output directory for assets
os.makedirs('assets', exist_ok=True)
a0 = 1.0  # Bohr radius

def R_nl(r, n, l):
    """Radial wave function for Hydrogen."""
    rho = 2.0 * r / (n * a0)
    norm = np.sqrt((2.0 / (n * a0))**3 * math.factorial(n - l - 1) / (2.0 * n * math.factorial(n + l)))
    laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return norm * np.exp(-rho / 2.0) * rho**l * laguerre

def real_sph_harm(l, m, theta, phi):
    """Computes the Real Spherical Harmonic."""
    Y_c = sph_harm_y(l, abs(m), phi, theta)
    if m < 0:
        return np.sqrt(2) * (-1)**m * Y_c.imag
    elif m > 0:
        return np.sqrt(2) * (-1)**m * Y_c.real
    else:
        return Y_c.real

def probability_density(r, theta, phi, n, l, m):
    """Vacuum density field representing the 'Pilot Wave' pressure map."""
    R = R_nl(r, n, l)
    Y = real_sph_harm(l, m, theta, phi)
    val = np.clip(R * Y, -1e100, 1e100)
    return val**2

def generate_reference_cloud(n, l, m, num_samples=100000):
    """
    Generates a true probability density cloud using Monte Carlo rejection sampling
    (the standard Schrödinger method) to serve as a reference.
    """
    r_max = n**2 * a0 * 3.0
    batch_size = num_samples * 10
    samples_r = np.array([])
    samples_theta = np.array([])
    samples_phi = np.array([])

    # Find empirical max density
    r_test = np.linspace(0.1, r_max, 100)
    theta_test = np.linspace(0, np.pi, 50)
    R, T = np.meshgrid(r_test, theta_test)
    probs = probability_density(R, T, 0, n, l, m) * (R**2)
    max_density = np.max(probs) * 1.5

    while len(samples_r) < num_samples:
        r = np.random.uniform(0, r_max, batch_size)
        theta = np.arccos(1 - 2 * np.random.uniform(0, 1, batch_size))
        phi = np.random.uniform(0, 2 * np.pi, batch_size)

        prob = probability_density(r, theta, phi, n, l, m) * (r**2)
        accepted = np.random.uniform(0, max_density, batch_size) < prob

        samples_r = np.append(samples_r, r[accepted])
        samples_theta = np.append(samples_theta, theta[accepted])
        samples_phi = np.append(samples_phi, phi[accepted])

    samples_r = samples_r[:num_samples]
    samples_theta = samples_theta[:num_samples]
    samples_phi = samples_phi[:num_samples]

    x = samples_r * np.sin(samples_theta) * np.cos(samples_phi)
    y = samples_r * np.sin(samples_theta) * np.sin(samples_phi)
    z = samples_r * np.cos(samples_theta)

    fig = plt.figure(figsize=(4, 4), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')

    colors = ['cyan', 'lime', 'magenta', 'yellow', 'orange', 'white']
    color = colors[l % len(colors)]

    ax.scatter(x, y, z, c=color, s=0.5, alpha=0.1, edgecolors='none')

    max_val = np.max(np.abs([x, y, z])) * 0.8
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.axis('off')

    # Title indicating this is standard probability
    ax.set_title(f'Schrödinger Probability: {n}{"spdfgh"[l]} (m={m})', color='white', fontsize=10)

    safe_m = str(m).replace('-', 'minus_')
    filename = f'assets/ref_{n}_{l}_{safe_m}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close()
    return filename

if __name__ == "__main__":
    np.random.seed(42)
    states = [
        ("1s", 1, 0, 0),
        ("2s", 2, 0, 0),
        ("2p0", 2, 1, 0),
        ("2p1", 2, 1, 1),
        ("2p-1", 2, 1, -1),
        ("3s", 3, 0, 0),
        ("3p0", 3, 1, 0),
        ("3d-2", 3, 2, -2),
        ("3d-1", 3, 2, -1),
        ("3d0", 3, 2, 0),
        ("3d1", 3, 2, 1),
        ("3d2", 3, 2, 2),
        ("4s", 4, 0, 0),
        ("4d-2", 4, 2, -2),
        ("4d-1", 4, 2, -1),
        ("4d0", 4, 2, 0),
        ("4d1", 4, 2, 1),
        ("4d2", 4, 2, 2),
        ("4f-3", 4, 3, -3),
        ("4f-2", 4, 3, -2),
        ("4f-1", 4, 3, -1),
        ("4f0", 4, 3, 0),
        ("4f1", 4, 3, 1),
        ("4f2", 4, 3, 2),
        ("4f3", 4, 3, 3),
        ("5s", 5, 0, 0),
        ("5d2", 5, 2, 2),
        ("5f3", 5, 3, 3),
        ("5g4", 5, 4, 4),
    ]

    print("Generating standard probability clouds for reference...")
    for name, n, l, m in states:
        print(f"Generating {name}...")
        generate_reference_cloud(n, l, m, num_samples=30000)
    print("Done.")
