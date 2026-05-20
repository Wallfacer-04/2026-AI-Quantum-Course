"""QSVT Quantum PCA on Rydberg Atoms.

Demonstrates eigenvalue filtering for principal component analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from rydberg_qsvt.rydberg_model import create_rydberg_device
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit
from rydberg_qsvt.utils import save_figure


def run_qpca(n_qubits: int = 2, threshold: float = 0.5):
    """Run quantum PCA and return filtered eigenvalue distribution."""
    eigenvalues = np.array([0.5, 0.3, 0.15, 0.05])

    degrees = [4, 8, 12, 16]
    filtered_states = []

    for degree in degrees:
        coeffs = [1.0 if i % 2 == 0 else 0.0 for i in range(degree)]
        phases = compute_phase_angles(coeffs, degree=degree)

        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)

        mask = eigenvalues > threshold
        filtered = probs[: len(eigenvalues)] * mask
        filtered = filtered / (filtered.sum() + 1e-10)
        filtered_states.append(filtered)

    return eigenvalues, degrees, filtered_states


def plot_qpca(eigenvalues, degrees, filtered_states):
    """Plot QPCA results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(len(eigenvalues)), eigenvalues)
    axes[0].axhline(y=0.5, color="r", linestyle="--", label="Threshold")
    axes[0].set_xlabel("Eigenvalue Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Original Density Matrix Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    width = 0.8 / len(degrees)
    for i, (degree, filtered) in enumerate(zip(degrees, filtered_states)):
        offset = (i - len(degrees) / 2) * width
        axes[1].bar(
            np.arange(len(filtered)) + offset, filtered, width, label=f"d={degree}"
        )
    axes[1].set_xlabel("State Index")
    axes[1].set_ylabel("Filtered Probability")
    axes[1].set_title("QSVT-Filtered Principal Components")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    save_figure(fig, "qpca")


if __name__ == "__main__":
    eigenvalues, degrees, filtered_states = run_qpca()
    plot_qpca(eigenvalues, degrees, filtered_states)
    print("Quantum PCA complete. Figures saved to figures/")
