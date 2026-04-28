import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift

os.makedirs("assets/deconvolution", exist_ok=True)

targets = {
    "biotin": "Biotin Molecular Crystal",
    "nidppfcl2": "Ni(dppf)Cl2",
    "graphene": "Monolayer Graphene",
    "srtio3": "Strontium Titanate (SrTiO3)"
}

def create_base_lattice(target_id, N=512):
    x = np.linspace(-10, 10, N)
    y = np.linspace(-10, 10, N)
    X, Y = np.meshgrid(x, y)

    if target_id == "biotin":
        # Organic hydrogen bond network
        Z = np.sin(X*3)*np.cos(Y*3) + 0.5*np.exp(-(X**2+Y**2)/5)
    elif target_id == "nidppfcl2":
        # Complex organometallic
        Z = np.cos(X*2)**2 * np.sin(Y*2)**2 + 0.5*np.cos(X*4+Y*4)
    elif target_id == "graphene":
        # Hexagonal lattice
        Z = np.cos(X) + np.cos(X*0.5 + Y*0.866) + np.cos(X*0.5 - Y*0.866)
    else: # srtio3
        # Perovskite structure (heavy and light columns)
        Z = np.exp(-(X%2)**2 - (Y%2)**2) + 0.3*np.exp(-((X-1)%2)**2 - ((Y-1)%2)**2)

    # Normalize
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z

def geon_psf(N=512):
    kx = np.linspace(-5, 5, N)
    ky = np.linspace(-5, 5, N)
    KX, KY = np.meshgrid(kx, ky)

    r = np.sqrt(KX**2 + KY**2)
    theta = np.arctan2(KY, KX)

    A = 0.8
    # e^{-r^2/2} * [1 + A*cos(2\theta)*e^{-r}]
    psf = np.exp(-r**2/2) * (1 + A * np.cos(2*theta) * np.exp(-r))
    return psf / psf.sum()

def process_target(target_id):
    N = 512
    base = create_base_lattice(target_id, N)

    # Generate PSF
    psf = geon_psf(N)

    # Convolution to simulate raw blurred microscope image (astigmatism)
    base_fft = fft2(base)
    psf_fft = fft2(ifftshift(psf))
    raw_img = np.real(ifft2(base_fft * psf_fft))

    # Add ZPF noise
    np.random.seed(42)
    raw_img += np.random.normal(0, 0.05, (N, N))
    raw_img = np.clip(raw_img, 0, 1)

    # A. Control Image (Standard CoM/VBF with typical blur)
    # We use the raw_img to represent what standard STEM CoM shows
    control = raw_img.copy()

    # B. Experimental Image (Geon Deconvolution)
    epsilon = 1e-4
    raw_fft = fft2(raw_img)
    # Wiener filter deterministic inversion
    deconv_fft = raw_fft / (psf_fft + epsilon)
    experimental = np.real(ifft2(deconv_fft))

    # Normalize for display
    control = (control - control.min()) / (control.max() - control.min())
    experimental = (experimental - experimental.min()) / (experimental.max() - experimental.min())

    # Save
    plt.imsave(f"assets/deconvolution/{target_id}_control.png", control, cmap="magma")
    plt.imsave(f"assets/deconvolution/{target_id}_experimental.png", experimental, cmap="magma")

def main():
    for target in targets.keys():
        print(f"Processing {target}...")
        process_target(target)
    print("All targets processed successfully.")

if __name__ == "__main__":
    main()
