import numpy as np
import pennylane as qml
from rydberg_qsvt.rydberg_model import (
    rydberg_hamiltonian,
    rydberg_blockade_radius,
    create_rydberg_device,
)
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit


def test_rydberg_hamiltonian_shape():
    """Rydberg Hamiltonian for 2 qubits should be 4x4."""
    n_qubits = 2
    H = rydberg_hamiltonian(n_qubits, omega=1.0, delta=0.0, C6=1.0)
    assert H.shape == (4, 4)
    # Hamiltonian should be Hermitian
    np.testing.assert_allclose(H, H.conj().T, atol=1e-10)


def test_rydberg_blockade_radius():
    """Blockade radius should scale as (C6/delta)^(1/6)."""
    r = rydberg_blockade_radius(C6=1.0, delta=1.0)
    np.testing.assert_allclose(r, 1.0, atol=1e-10)


def test_create_rydberg_device():
    """Device should support at least 3 wires."""
    dev = create_rydberg_device(n_qubits=3)
    assert len(dev.wires) >= 3


def test_compute_phase_angles_constant():
    """Constant polynomial P(x) = 1 should return valid phase angles."""
    phases = compute_phase_angles(coefficients=[1.0])
    assert isinstance(phases, list)
    assert len(phases) > 0


def test_qsvt_circuit_returns_counts():
    """QSVT circuit should return measurement results."""
    dev = qml.device("default.qubit", wires=2)
    result = build_qsvt_circuit(dev, phases=[0.0, 0.0, 0.0], target_wire=0)
    assert result is not None


def test_hamiltonian_simulation_runs():
    """Hamiltonian simulation should complete without error."""
    from rydberg_qsvt.app_hamiltonian_sim import run_hamiltonian_simulation

    times, fidelities = run_hamiltonian_simulation(n_qubits=2, t_max=1.0, n_steps=5)
    assert len(times) == 5
    assert len(fidelities) == 5
    assert all(0 <= f <= 1 for f in fidelities)


def test_linear_system_runs():
    """Linear system solver should complete without error."""
    from rydberg_qsvt.app_linear_system import run_linear_system_solver

    degrees, fidelities, x_exact = run_linear_system_solver(n_qubits=2)
    assert len(degrees) == len(fidelities)
    assert len(x_exact) == 4


def test_qpca_runs():
    """QPCA should complete without error."""
    from rydberg_qsvt.app_qpca import run_qpca

    eigenvalues, degrees, filtered_states = run_qpca(n_qubits=2)
    assert len(eigenvalues) == 4
    assert len(filtered_states) == len(degrees)
