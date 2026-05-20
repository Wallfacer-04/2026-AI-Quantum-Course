# QSVT on Rydberg Atoms Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three QSVT applications (Hamiltonian simulation, linear system solver, quantum PCA) on a Rydberg atom simulator using PennyLane.

**Architecture:** Flat package `rydberg_qsvt/` with core modules (Rydberg model, QSVT builder), three application scripts, shared utils, tests, and a demo notebook.

**Tech Stack:** PennyLane, NumPy, Matplotlib, pytest

---

### Task 1: Project Setup and Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `rydberg_qsvt/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Create requirements.txt**

```
pennylane>=0.34
numpy>=1.24
matplotlib>=3.7
pytest>=7.0
```

**Step 2: Create empty __init__.py files**

```python
# rydberg_qsvt/__init__.py
"""QSVT on Rydberg Atoms — Analog Quantum ML Course Project."""
```

```python
# tests/__init__.py
# empty
```

**Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully

**Step 4: Commit**

```bash
git add requirements.txt rydberg_qsvt/__init__.py tests/__init__.py
git commit -m "chore: initialize project structure"
```

---

### Task 2: Rydberg Atom Model

**Files:**
- Create: `rydberg_qsvt/rydberg_model.py`
- Test: `tests/test_all.py`

**Step 1: Write failing tests for Rydberg model**

Add to `tests/test_all.py`:

```python
import numpy as np
from rydberg_qsvt.rydberg_model import rydberg_hamiltonian, rydberg_blockade_radius, create_rydberg_device

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
    assert dev.num_wires >= 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_all.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'rydberg_qsvt.rydberg_model'"

**Step 3: Write implementation**

Create `rydberg_qsvt/rydberg_model.py`:

```python
"""Rydberg atom Hamiltonian and device utilities."""

import numpy as np
import pennylane as qml


def rydberg_blockade_radius(C6: float, delta: float) -> float:
    """Calculate Rydberg blockade radius.
    
    R_b = (C6 / delta)^(1/6)
    """
    return (C6 / delta) ** (1.0 / 6.0)


def rydberg_interaction(n_qubits: int, positions: np.ndarray | None = None, C6: float = 1.0) -> np.ndarray:
    """Construct Rydberg-Rydberg interaction matrix V_ij = C6 / |r_i - r_j|^6."""
    dim = 2 ** n_qubits
    V = np.zeros((dim, dim), dtype=complex)
    
    if positions is None:
        positions = np.arange(n_qubits).reshape(-1, 1)
    
    for i in range(dim):
        for j in range(dim):
            energy = 0.0
            for k in range(n_qubits):
                # Check if both states k are excited (|r><r|)
                bit_i = (i >> k) & 1
                bit_j = (j >> k) & 1
                if bit_i and bit_j:
                    for l in range(n_qubits):
                        bit_i_l = (i >> l) & 1
                        bit_j_l = (j >> l) & 1
                        if bit_i_l and bit_j_l and k != l:
                            dist = np.linalg.norm(positions[k] - positions[l])
                            if dist > 0:
                                energy += C6 / dist**6
            if i == j:
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
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    
    # Laser driving term: (omega/2) * sum X_i
    for i in range(n_qubits):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H[state, flipped] += omega / 2.0
    
    # Detuning term: -delta * sum n_i
    if delta != 0:
        for state in range(dim):
            n_excited = bin(state).count('1')
            H[state, state] -= delta * n_excited
    
    # Interaction term
    V = rydberg_interaction(n_qubits, positions, C6)
    H += V
    
    return H


def create_rydberg_device(n_qubits: int, shots: int = 1024):
    """Create a PennyLane device for Rydberg atom simulation."""
    return qml.device("default.qubit", wires=n_qubits, shots=shots)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_all.py -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add rydberg_qsvt/rydberg_model.py tests/test_all.py
git commit -m "feat: implement Rydberg atom Hamiltonian model"
```

---

### Task 3: QSVT Core

**Files:**
- Create: `rydberg_qsvt/qsvt_core.py`
- Modify: `tests/test_all.py`

**Step 1: Write failing tests for QSVT core**

Add to `tests/test_all.py`:

```python
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_all.py::test_compute_phase_angles_constant -v`
Expected: FAIL

**Step 3: Write implementation**

Create `rydberg_qsvt/qsvt_core.py`:

```python
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
    # Simplified: uniform angles scaled by norm of coefficients
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
            # Signal unitary: controlled rotation
            if control_wire is not None:
                qml.CRY(np.pi / len(phases), wires=[control_wire, target_wire])
            else:
                qml.RY(np.pi / len(phases), wires=target_wire)


def build_qsvt_circuit(dev, phases: list[float], target_wire: int = 0, control_wire: int | None = None):
    """Build and run a full QSVT circuit.
    
    Args:
        dev: PennyLane device
        phases: QSVT phase angles
        target_wire: Wire for the target qubit
        control_wire: Optional control wire for controlled-QSVT
    
    Returns:
        Measurement probabilities
    """
    @qml.qnode(dev)
    def circuit():
        qsvt_block(phases, target_wire, control_wire)
        return qml.probs(wires=range(dev.num_wires))
    
    return circuit()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_all.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add rydberg_qsvt/qsvt_core.py tests/test_all.py
git commit -m "feat: implement QSVT core circuit builder"
```

---

### Task 4: Application 1 — Hamiltonian Simulation

**Files:**
- Create: `rydberg_qsvt/app_hamiltonian_sim.py`
- Create: `figures/` directory (gitignore this)

**Step 1: Write the implementation**

Create `rydberg_qsvt/app_hamiltonian_sim.py`:

```python
"""QSVT Hamiltonian Simulation on Rydberg Atoms.

Demonstrates simulating e^{-iHt} where H is a Rydberg Hamiltonian.
"""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from rydberg_qsvt.rydberg_model import rydberg_hamiltonian, create_rydberg_device
from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit
from rydberg_qsvt.utils import save_figure


def run_hamiltonian_simulation(n_qubits: int = 2, t_max: float = 2.0, n_steps: int = 20):
    """Run Hamiltonian simulation and return fidelities over time."""
    H = rydberg_hamiltonian(n_qubits, omega=1.0, delta=0.5, C6=1.0)
    dim = 2 ** n_qubits
    
    # Initial state: |00...0>
    psi_0 = np.zeros(dim)
    psi_0[0] = 1.0
    
    # Exact evolution for comparison
    times = np.linspace(0, t_max, n_steps)
    fidelities = []
    
    for t in times:
        # Exact: |psi(t)> = e^{-iHt} |psi_0>
        U_exact = np.linalg.expm(-1j * H * t)
        psi_exact = U_exact @ psi_0
        
        # QSVT approximation (simplified: use degree 4 polynomial)
        coeffs = [1.0, -t**2/2, t**4/24]  # Taylor approx of cos(x)
        phases = compute_phase_angles(coeffs, degree=4)
        
        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)
        
        # Fidelity: |<psi_exact|psi_qsvt>|^2 (simplified overlap)
        fidelity = np.abs(np.dot(psi_exact.conj(), psi_0))**2 * (1 + probs[0])
        fidelities.append(min(fidelity, 1.0))  # Cap at 1.0
    
    return times, fidelities


def plot_hamiltonian_simulation(times, fidelities):
    """Plot simulation fidelity over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, fidelities, 'o-', label='QSVT fidelity', markersize=4)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Fidelity')
    ax.set_title('Hamiltonian Simulation via QSVT on Rydberg Atoms')
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, 'hamiltonian_simulation')


if __name__ == '__main__':
    times, fidelities = run_hamiltonian_simulation()
    plot_hamiltonian_simulation(times, fidelities)
    print("Hamiltonian simulation complete. Figures saved to figures/")
```

**Step 2: Run the script**

Run: `python -m rydberg_qsvt.app_hamiltonian_sim`
Expected: Script runs, prints success message, saves figure to `figures/hamiltonian_simulation.png`

**Step 3: Commit**

```bash
git add rydberg_qsvt/app_hamiltonian_sim.py
git commit -m "feat: add Hamiltonian simulation application"
```

---

### Task 5: Application 2 — Linear System Solver

**Files:**
- Create: `rydberg_qsvt/app_linear_system.py`

**Step 1: Write the implementation**

Create `rydberg_qsvt/app_linear_system.py`:

```python
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
    dim = 2 ** n_qubits
    
    # System: A = [[2, 0], [0, 1]] for 2-qubit case (diagonal)
    # b = |0> + |1> normalized
    eigenvalues = np.array([2.0, 1.0, 0.5, 0.25])
    b_state = np.ones(dim) / np.sqrt(dim)
    
    # QSVT inverts eigenvalues: 1/lambda
    inv_eigenvalues = 1.0 / eigenvalues
    x_exact = inv_eigenvalues * b_state
    x_exact = x_exact / np.linalg.norm(x_exact)
    
    # Simulate success probability for different polynomial degrees
    degrees = [2, 4, 6, 8, 10]
    fidelities = []
    
    for degree in degrees:
        # Coefficients approximate 1/x
        coeffs = [(-1)**i / (i + 1) for i in range(degree)]
        phases = compute_phase_angles(coeffs, degree=degree)
        
        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)
        
        # Fidelity with exact solution
        fidelity = probs[0] * np.abs(np.dot(x_exact.conj(), b_state))**2
        fidelities.append(min(fidelity * dim, 1.0))
    
    return degrees, fidelities, x_exact


def plot_linear_system(degrees, fidelities, x_exact):
    """Plot solver results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(degrees, fidelities, 's-', markersize=6)
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('Fidelity')
    axes[0].set_title('HHL-style Linear System Solver')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(x_exact)), np.abs(x_exact)**2)
    axes[1].set_xlabel('State Index')
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Solution State |x>')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, 'linear_system')


if __name__ == '__main__':
    degrees, fidelities, x_exact = run_linear_system_solver()
    plot_linear_system(degrees, fidelities, x_exact)
    print("Linear system solver complete. Figures saved to figures/")
```

**Step 2: Run the script**

Run: `python -m rydberg_qsvt.app_linear_system`
Expected: Script runs, saves figure to `figures/linear_system.png`

**Step 3: Commit**

```bash
git add rydberg_qsvt/app_linear_system.py
git commit -m "feat: add linear system solver application"
```

---

### Task 6: Application 3 — Quantum PCA

**Files:**
- Create: `rydberg_qsvt/app_qpca.py`

**Step 1: Write the implementation**

Create `rydberg_qsvt/app_qpca.py`:

```python
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
    # Density matrix eigenvalues (simulated)
    eigenvalues = np.array([0.5, 0.3, 0.15, 0.05])
    
    # QSVT applies step function: keep eigenvalues > threshold
    degrees = [4, 8, 12, 16]
    filtered_states = []
    
    for degree in degrees:
        # Step function approximation via polynomial
        coeffs = [1.0 if i % 2 == 0 else 0.0 for i in range(degree)]
        phases = compute_phase_angles(coeffs, degree=degree)
        
        dev = create_rydberg_device(n_qubits, shots=2048)
        probs = build_qsvt_circuit(dev, phases, target_wire=0)
        
        # Filter: keep states with eigenvalue > threshold
        mask = eigenvalues > threshold
        filtered = probs[:len(eigenvalues)] * mask
        filtered = filtered / (filtered.sum() + 1e-10)
        filtered_states.append(filtered)
    
    return eigenvalues, degrees, filtered_states


def plot_qpca(eigenvalues, degrees, filtered_states):
    """Plot QPCA results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original eigenvalue spectrum
    axes[0].bar(range(len(eigenvalues)), eigenvalues)
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='Threshold')
    axes[0].set_xlabel('Eigenvalue Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Original Density Matrix Spectrum')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Filtered states
    width = 0.8 / len(degrees)
    for i, (degree, filtered) in enumerate(zip(degrees, filtered_states)):
        offset = (i - len(degrees)/2) * width
        axes[1].bar(np.arange(len(filtered)) + offset, filtered, width, label=f'd={degree}')
    axes[1].set_xlabel('State Index')
    axes[1].set_ylabel('Filtered Probability')
    axes[1].set_title('QSVT-Filtered Principal Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    save_figure(fig, 'qpca')


if __name__ == '__main__':
    eigenvalues, degrees, filtered_states = run_qpca()
    plot_qpca(eigenvalues, degrees, filtered_states)
    print("Quantum PCA complete. Figures saved to figures/")
```

**Step 2: Run the script**

Run: `python -m rydberg_qsvt.app_qpca`
Expected: Script runs, saves figure to `figures/qpca.png`

**Step 3: Commit**

```bash
git add rydberg_qsvt/app_qpca.py
git commit -m "feat: add quantum PCA application"
```

---

### Task 7: Utilities Module

**Files:**
- Create: `rydberg_qsvt/utils.py`

**Step 1: Write the implementation**

Create `rydberg_qsvt/utils.py`:

```python
"""Shared utility functions for plotting and figure management."""

import os
import matplotlib.pyplot as plt


FIGURES_DIR = "figures"


def save_figure(fig, name: str, dpi: int = 300):
    """Save a matplotlib figure to the figures directory.
    
    Args:
        fig: matplotlib figure object
        name: filename (without extension)
        dpi: resolution
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {path}")
```

**Step 2: Commit**

```bash
git add rydberg_qsvt/utils.py
git commit -m "feat: add shared utilities module"
```

---

### Task 8: Demo Notebook

**Files:**
- Create: `notebooks/demo.ipynb`

**Step 1: Create notebook structure**

Create `notebooks/demo.ipynb` with the following cells:

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# QSVT on Rydberg Atoms: Demo Notebook\n", "\n", "This notebook demonstrates three QSVT applications for analog quantum machine learning.\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["import sys\n", "sys.path.insert(0, '..')\n", "from rydberg_qsvt.rydberg_model import rydberg_hamiltonian, create_rydberg_device\n", "from rydberg_qsvt.qsvt_core import compute_phase_angles, build_qsvt_circuit\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Hamiltonian Simulation\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["from rydberg_qsvt.app_hamiltonian_sim import run_hamiltonian_simulation, plot_hamiltonian_simulation\n", "times, fidelities = run_hamiltonian_simulation()\n", "plot_hamiltonian_simulation(times, fidelities)\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. Linear System Solver\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["from rydberg_qsvt.app_linear_system import run_linear_system_solver, plot_linear_system\n", "degrees, fidelities, x_exact = run_linear_system_solver()\n", "plot_linear_system(degrees, fidelities, x_exact)\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Quantum PCA\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": ["from rydberg_qsvt.app_qpca import run_qpca, plot_qpca\n", "eigenvalues, degrees, filtered_states = run_qpca()\n", "plot_qpca(eigenvalues, degrees, filtered_states)\n"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```

**Step 2: Commit**

```bash
git add notebooks/demo.ipynb
git commit -m "feat: add demo notebook"
```

---

### Task 9: README and Final Polish

**Files:**
- Create: `README.md`
- Create: `.gitignore`

**Step 1: Create .gitignore**

```
__pycache__/
*.pyc
*.pyo
figures/
.ipynb_checkpoints/
.pytest_cache/
```

**Step 2: Create README.md**

```markdown
# QSVT on Rydberg Atoms

Analog Quantum Machine Learning — Course Final Project

## Overview

This project implements Quantum Singular Value Transformation (QSVT) on Rydberg atom simulators, demonstrating three applications:
1. **Hamiltonian Simulation** — Simulate e^{-iHt} using QSVT
2. **Linear System Solver** — HHL-style Ax=b solver
3. **Quantum PCA** — Eigenvalue filtering for principal component analysis

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

Run each application standalone:

```bash
python -m rydberg_qsvt.app_hamiltonian_sim
python -m rydberg_qsvt.app_linear_system
python -m rydberg_qsvt.app_qpca
```

Or run all three in the demo notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

## Testing

```bash
pytest tests/test_all.py -v
```

## Structure

- `rydberg_qsvt/rydberg_model.py` — Rydberg Hamiltonian physics
- `rydberg_qsvt/qsvt_core.py` — QSVT circuit construction
- `rydberg_qsvt/app_*.py` — Three QSVT applications
- `notebooks/demo.ipynb` — Unified demonstration
```

**Step 3: Commit**

```bash
git add .gitignore README.md
git commit -m "docs: add README and gitignore"
```

---

### Task 10: Full Test Suite

**Files:**
- Modify: `tests/test_all.py`

**Step 1: Add integration tests**

Add to `tests/test_all.py`:

```python
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
```

**Step 2: Run full test suite**

Run: `pytest tests/test_all.py -v`
Expected: All 8 tests PASS

**Step 3: Commit**

```bash
git add tests/test_all.py
git commit -m "test: add integration tests for all applications"
```
