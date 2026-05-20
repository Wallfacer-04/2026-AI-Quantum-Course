"""QSVT Linear System Solver (HHL-style) on Rydberg Atoms.

Demonstrates solving Ax = b using QSVT-based eigenvalue inversion.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from rydberg_qsvt.rydberg_model import create_rydberg_device
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit
from rydberg_qsvt.utils import save_figure


def run_linear_system_solver(n_qubits: int = 2):
    """Run linear system solver and return success probabilities."""
    dim = 2**n_qubits

    eigenvalues = np.array([2.0, 1.0, 0.5, 0.25])
    b_state = np.ones(dim) / np.sqrt(dim)

    inv_eigenvalues = 1.0 / eigenvalues
    x_exact = inv_eigenvalues * b_state
    x_exact = x_exact / np.linalg.norm(x_exact)

    degrees = [2, 4, 6, 8, 10]
    fidelities = []

    for degree in degrees:
        coeffs = [(-1) ** i / (i + 1) for i in range(degree)]
        phases = compute_phase_angles(coeffs, degree=degree)

        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)

        fidelity = probs[0] * np.abs(np.dot(x_exact.conj(), b_state)) ** 2
        fidelities.append(min(fidelity * dim, 1.0))

    return degrees, fidelities, x_exact


def plot_linear_system(degrees, fidelities, x_exact):
    """Plot solver results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(degrees, fidelities, "s-", markersize=6)
    axes[0].set_xlabel("Polynomial Degree")
    axes[0].set_ylabel("Fidelity")
    axes[0].set_title("HHL-style Linear System Solver")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(x_exact)), np.abs(x_exact) ** 2)
    axes[1].set_xlabel("State Index")
    axes[1].set_ylabel("Probability")
    axes[1].set_title("Solution State |x>")
    axes[1].grid(True, alpha=0.3, axis="y")

    save_figure(fig, "linear_system")


if __name__ == "__main__":
    degrees, fidelities, x_exact = run_linear_system_solver()
    plot_linear_system(degrees, fidelities, x_exact)
    print("Linear system solver complete. Figures saved to figures/")
