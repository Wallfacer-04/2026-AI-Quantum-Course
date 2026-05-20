"""Rydberg atom Hamiltonian and device utilities."""

import numpy as np
import pennylane as qml


def rydberg_blockade_radius(C6: float, delta: float) -> float:
    """Calculate Rydberg blockade radius.

    R_b = (C6 / delta)^(1/6)
    """
    return (C6 / delta) ** (1.0 / 6.0)


def rydberg_interaction(
    n_qubits: int, positions: np.ndarray | None = None, C6: float = 1.0
) -> np.ndarray:
    """Construct Rydberg-Rydberg interaction matrix V_ij = C6 / |r_i - r_j|^6."""
    dim = 2**n_qubits
    V = np.zeros((dim, dim), dtype=complex)

    if positions is None:
        positions = np.arange(n_qubits).reshape(-1, 1)

    for i in range(dim):
        excited = [k for k in range(n_qubits) if (i >> k) & 1]
        energy = 0.0
        for idx, k in enumerate(excited):
            for l in excited[idx + 1 :]:
                dist = np.linalg.norm(positions[k] - positions[l])
                if dist > 0:
                    energy += C6 / dist**6
        V[i, i] = energy

    return V


def rydberg_hamiltonian(
    n_qubits: int,
    omega: float = 1.0,
    delta: float = 0.0,
    C6: float = 1.0,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Construct full Rydberg Hamiltonian.

    H = (omega/2) * sum_i X_i - delta * sum_i n_i + sum_{i<j} V_ij n_i n_j

    Returns a 2^n x 2^n matrix.
    """
    dim = 2**n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # Laser driving term: (omega/2) * sum X_i
    for i in range(n_qubits):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += omega / 2.0

    # Detuning term: -delta * sum n_i
    if delta != 0:
        for state in range(dim):
            n_excited = bin(state).count("1")
            H[state, state] -= delta * n_excited

    # Interaction term
    V = rydberg_interaction(n_qubits, positions, C6)
    H += V

    return H


def create_rydberg_device(n_qubits: int, shots: int = 1024):
    """Create a PennyLane device for Rydberg atom simulation."""
    return qml.device("default.qubit", wires=n_qubits, shots=shots)
