#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt-get update -y

# Install Python and pip if not already installed
echo "Installing Python3 and pip..."
sudo apt-get install -y python3 python3-pip

# Install LaTeX dependencies for compiling the paper
echo "Installing LaTeX dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-extra-utils texlive-latex-extra texlive-bibtex-extra bzip2

# Install necessary Python packages for scientific computing and plotting
echo "Installing scientific computing packages (numpy, scipy, matplotlib)..."
pip3 install numpy scipy matplotlib

# Make the python script executable (optional but good practice)
chmod +x simulate_orbitals.py

# Run the simulation
echo "Running the orbital simulation..."
python3 simulate_orbitals.py

echo "Simulation complete! Check the orbitals_report.md and orbital_plots directory for results."
