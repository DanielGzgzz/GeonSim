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
        Z = np.sin(X*3)*np.cos(Y*3) + 0.5*np.exp(-(X**2+Y**2)/5)
    elif target_id == "nidppfcl2":
        Z = np.cos(X*2)**2 * np.sin(Y*2)**2 + 0.5*np.cos(X*4+Y*4)
    elif target_id == "graphene":
        Z = np.cos(X) + np.cos(X*0.5 + Y*0.866) + np.cos(X*0.5 - Y*0.866)
    else: # srtio3
        Z = np.exp(-(X%2)**2 - (Y%2)**2) + 0.3*np.exp(-((X-1)%2)**2 - ((Y-1)%2)**2)

    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z

def geon_psf(N=512, phi=np.pi/6):
    """
    FIX 1 & 2: Shifted coordinates and phase angle phi.
    Creates PSF_geon with centered frequencies, then ifftshifts it to match scipy.fft origin.
    """
    # Create frequencies correctly centered for the grid
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    KX, KY = np.meshgrid(kx, ky)

    # We must shift them to visualize/build symmetrically around (0,0) easily
    KX = fftshift(KX)
    KY = fftshift(KY)

    r = np.sqrt(KX**2 + KY**2)
    theta = np.arctan2(KY, KX)

    sigma = 100.0
    # Add phase phi for astigmatic rotational alignment.
    # Note: Using 4*theta to mimic standard quadrupolar astigmatism / 4-lobed star
    psf_centered = np.exp(-(r**2)/sigma) * (1 + 0.8 * np.cos(4 * theta + phi))

    psf_centered = psf_centered / psf_centered.sum()

    # Shift back to DC at (0,0) for FFT alignment
    return ifftshift(psf_centered)

def create_hann_window(N=512):
    """
    FIX 3: Low-Pass Window. Taper frequency domain edges to suppress ringing artifacts.
    """
    hann_1d = np.hanning(N)
    # 2D outer product
    hann_2d = np.outer(hann_1d, hann_1d)
    return ifftshift(hann_2d) # Shift to match FFT origin

def process_target(target_id):
    N = 512
    base = create_base_lattice(target_id, N)

    # Generate aligned PSF
    phi_val = np.pi/4 if target_id == "graphene" else np.pi/6 # slight angle differences
    psf_spatial = geon_psf(N, phi=phi_val)

    # Convolution
    base_fft = fft2(base)
    psf_fft = fft2(psf_spatial)

    raw_img = np.real(ifft2(base_fft * psf_fft))

    # Add noise
    np.random.seed(42)
    raw_img += np.random.normal(0, 0.05, (N, N))
    raw_img = np.clip(raw_img, 0, 1)

    control = raw_img.copy()

    # B. Experimental Image (Geon Deconvolution)
    epsilon = 1e-4
    raw_fft = fft2(raw_img)

    # Wiener filter deterministic inversion
    deconv_fft = raw_fft / (psf_fft + epsilon)

    # Apply FIX 3: Hann Window Low-Pass
    hann_window = create_hann_window(N)
    deconv_fft = deconv_fft * hann_window

    experimental = np.real(ifft2(deconv_fft))

    # Normalize for display
    control = (control - control.min()) / (control.max() - control.min())
    experimental = (experimental - experimental.min()) / (experimental.max() - experimental.min())

    plt.imsave(f"assets/deconvolution/{target_id}_control.png", control, cmap="magma")
    plt.imsave(f"assets/deconvolution/{target_id}_experimental.png", experimental, cmap="magma")

def main():
    for target in targets.keys():
        print(f"Processing {target}...")
        process_target(target)
    print("All targets processed successfully with Fourier corrections.")

if __name__ == "__main__":
    main()
