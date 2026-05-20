"""QSVT circuit construction utilities."""

import numpy as np
import pennylane as qml


def compute_phase_angles(coefficients: list[float], degree: int = 4) -> list[float]:
    """Compute QSVT phase angles for a target polynomial.

    Uses a simplified approximation: for small degree polynomials,
    returns evenly spaced phase angles scaled by coefficients.

    In practice, these are found via optimization or the Remez algorithm.
    """
    n_angles = degree + 1
    norm = np.sqrt(sum(c**2 for c in coefficients))
    base_angle = np.arctan(norm) / n_angles if norm > 0 else 0.0
    return [base_angle * (i + 1) for i in range(n_angles)]


def projector_rotation(wire: int, phi: float):
    """Apply a single projector-controlled rotation R_z(phi) with X gates."""
    qml.RX(2 * phi, wires=wire)


def qsvt_block(phases: list[float], target_wire: int, control_wire: int | None = None):
    """Apply a QSVT sequence: alternating signal unitary and projector rotations.

    U(phi_0) A U(phi_1) A ... U(phi_d)

    where A is the signal unitary (here, a simple rotation) and U(phi) are phase rotations.
    """
    for i, phi in enumerate(phases):
        projector_rotation(target_wire, phi)
        if i < len(phases) - 1:
            if control_wire is not None:
                qml.CRY(np.pi / len(phases), wires=[control_wire, target_wire])
            else:
                qml.RY(np.pi / len(phases), wires=target_wire)


def build_qsvt_circuit(
    dev, phases: list[float], target_wire: int = 0, control_wire: int | None = None
):
    """Build and run a full QSVT circuit."""

    @qml.qnode(dev)
    def circuit():
        qsvt_block(phases, target_wire, control_wire)
        return qml.probs(wires=range(len(dev.wires)))

    return circuit()
