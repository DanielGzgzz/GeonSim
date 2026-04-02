import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from mpl_toolkits.mplot3d import Axes3D

def real_sph_harm_correct(l, m, theta, phi):
    Y_c = sph_harm_y(l, abs(m), theta, phi)
    if m < 0:
        return np.sqrt(2) * Y_c.imag * (-1)**abs(m)
    elif m > 0:
        return np.sqrt(2) * Y_c.real * (-1)**abs(m)
    else:
        return Y_c.real

theta = np.linspace(0, np.pi, 100)
phi = np.linspace(0, 2*np.pi, 100)
theta, phi = np.meshgrid(theta, phi)

Y = real_sph_harm_correct(3, 1, theta, phi)
R = np.abs(Y)**2
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.savefig('test_3_1.png')

Y = real_sph_harm_correct(3, 0, theta, phi)
R = np.abs(Y)**2
x = R * np.sin(theta) * np.cos(phi)
y = R * np.sin(theta) * np.sin(phi)
z = R * np.cos(theta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.savefig('test_3_0.png')
