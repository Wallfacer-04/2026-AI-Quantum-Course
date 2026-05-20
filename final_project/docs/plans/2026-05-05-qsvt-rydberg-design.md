# QSVT on Rydberg Atoms — Design

> **Approved by user.** Simplified Approach C: standalone scripts in a flat package.

**Goal:** Demonstrate three QSVT applications (Hamiltonian simulation, linear system solver, quantum PCA) implemented on a 2-3 qubit Rydberg atom simulator.

**Stack:** PennyLane + NumPy + Matplotlib. Tests with pytest.

---

## Architecture

```
final_project/
├── rydberg_qsvt/
│   ├── __init__.py
│   ├── rydberg_model.py
│   ├── qsvt_core.py
│   ├── app_hamiltonian_sim.py
│   ├── app_linear_system.py
│   ├── app_qpca.py
│   └── utils.py
├── tests/
│   └── test_all.py
├── notebooks/
│   └── demo.ipynb
├── requirements.txt
└── README.md
```

## Data Flow

1. `rydberg_model.py` provides a PennyLane device and functions to construct Rydberg Hamiltonians
2. `qsvt_core.py` builds QSVT circuits: takes polynomial coefficients → returns phase angles → constructs circuit
3. Each `app_*.py` calls both, runs on `default.qubit` with Rydberg-like parameters, plots results
4. `utils.py` provides shared plotting functions

## Key Decisions

- 2-3 qubits throughout (fast simulation, clear visualization)
- Each app script is runnable standalone: `python -m rydberg_qsvt.app_*`
- Figures saved to `figures/` directory
- Notebook imports the apps for unified presentation
