import numpy as np
from simulate_orbitals import prob_density_cart

for n, l, m in [(1, 0, 0), (2, 1, 1), (3, 2, 2)]:
    max_p = 0
    for x in np.linspace(-10, 10, 50):
        for y in np.linspace(-10, 10, 50):
            for z in np.linspace(-10, 10, 50):
                p = prob_density_cart(x, y, z, n, l, m)
                if p > max_p:
                    max_p = p
    print(f"Max P for n={n}, l={l}, m={m}: {max_p}")
