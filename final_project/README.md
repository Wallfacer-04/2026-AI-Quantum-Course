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
