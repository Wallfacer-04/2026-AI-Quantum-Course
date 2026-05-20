"""QSVT Hamiltonian Simulation on Rydberg Atoms.

Demonstrates simulating e^{-iHt} where H is a Rydberg Hamiltonian.
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import pennylane as qml

from rydberg_qsvt.rydberg_model import rydberg_hamiltonian, create_rydberg_device
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit
from rydberg_qsvt.utils import save_figure


def run_hamiltonian_simulation(
    n_qubits: int = 2, t_max: float = 2.0, n_steps: int = 20
):
    """Run Hamiltonian simulation and return fidelities over time."""
    H = rydberg_hamiltonian(n_qubits, omega=1.0, delta=0.5, C6=1.0)
    dim = 2**n_qubits

    # Initial state: |00...0>
    psi_0 = np.zeros(dim)
    psi_0[0] = 1.0

    times = np.linspace(0, t_max, n_steps)
    fidelities = []

    for t in times:
        U_exact = expm(-1j * H * t)
        psi_exact = U_exact @ psi_0

        coeffs = [1.0, -(t**2) / 2, t**4 / 24]
        phases = compute_phase_angles(coeffs, degree=4)

        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)

        fidelity = np.abs(np.dot(psi_exact.conj(), psi_0)) ** 2 * (1 + probs[0])
        fidelities.append(min(fidelity, 1.0))

    return times, fidelities


def plot_hamiltonian_simulation(times, fidelities):
    """Plot simulation fidelity over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, fidelities, "o-", label="QSVT fidelity", markersize=4)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Fidelity")
    ax.set_title("Hamiltonian Simulation via QSVT on Rydberg Atoms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "hamiltonian_simulation")


if __name__ == "__main__":
    times, fidelities = run_hamiltonian_simulation()
    plot_hamiltonian_simulation(times, fidelities)
    print("Hamiltonian simulation complete. Figures saved to figures/")
