# BPA (Belief Poisoning Attack)

This repository provides an experimental framework for **Belief Poisoning Attack (BPA)**.

---

## Quick Start

### 1) Clone
```bash
git clone https://github.com/CoderWZW/BPA.git
cd BPA
```

### 2) Configure (edit `config.py`)
Open **`config.py`** and adjust the core parameters for your run.

---

## How to Run

### A) Baseline (no attack)
```bash
python run_experiment.py
```

### B) Attack
```bash
python run_experiment_attack.py
```

### C) Attack + Defense
```bash
python run_experiment_attack_defense.py
```
