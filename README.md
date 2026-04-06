# Quantum-Enhanced Continual Learning Fabric for Planetary Climate Adaptation

> Federated continual learning system for planetary climate forecasting — combining Physics-Informed Neural Networks, Elastic Weight Consolidation, and Multi-Agent RL across distributed geographic nodes.

---

## Overview

Most climate models are trained on centralized data. This project takes a different approach — training across **3 geographic regions simultaneously** (North America, Europe, Asia-Pacific) without ever moving raw data, while actively preventing the model from forgetting earlier climate patterns as distributions shift over time.

Built to production-grade standards with a full ablation suite, reproducible configs, and AWS SageMaker integration.

---

## Core Components

**Federated Learning (FedAvg)**
Regional nodes train locally. Only model weights are shared — never raw climate data.

**Continual Learning (EWC)**
Elastic Weight Consolidation prevents catastrophic forgetting as the model learns across time periods.

**Physics-Informed Neural Network (PINN)**
PDE residual losses enforce mass and energy conservation — the model cannot violate atmospheric physics.

**Multi-Agent RL (MARL)**
One PPO agent per region coordinates adaptive responses to local climate shifts.

---

## Ablation Matrix

| Condition            | Federated | EWC | PINN | MARL |
|----------------------|:---------:|:---:|:----:|:----:|
| Centralized baseline |     —     |  —  |  —   |  —   |
| + Federated          |     ✓     |  —  |  —   |  —   |
| + EWC                |     ✓     |  ✓  |  —   |  —   |
| + PINN               |     ✓     |  ✓  |  ✓   |  —   |
| Full model           |     ✓     |  ✓  |  ✓   |  ✓   |

---

## Project Structure
```text
├── src/
│   ├── data/           # ERA5 + synthetic data loaders
│   ├── models/         # PINN backbone and loss functions
│   ├── federated/      # FL server, clients, FedAvg, EWC
│   ├── marl/           # Multi-agent env, PPO agents
│   ├── evaluation/     # Metrics, forgetting benchmarks, ablation
│   └── utils/          # Config, logging, visualization
├── configs/            # YAML configs for all experiments
├── scripts/            # CLI entry points
├── notebooks/          # Exploration and results analysis
├── tests/              # Unit + integration tests
└── results/            # Checkpoints, figures, logs
```

---

## Quickstart
```bash
# Install
pip install -e .

# Run with synthetic data (no API key needed)
python scripts/run_experiment.py --config configs/experiment.yaml --synthetic

# Run full ablation suite
python scripts/run_ablation.py --output results/ablation_table.csv
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| [ERA5 Reanalysis](https://cds.climate.copernicus.eu/) | Hourly climate variables 1940–present |
| [CMIP6 Projections](https://esgf-node.llnl.gov/projects/cmip6/) | Future climate scenarios |
| Synthetic generator | Built-in, no API key required |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model training | PyTorch 2.0+ |
| Climate data | xarray, netCDF4 |
| Federated learning | FedAvg (custom) |
| Config management | OmegaConf |
| Cloud training | AWS SageMaker (Week 4+) |
| Experiment tracking | Weights & Biases |

---

## Key Metrics

| Metric | Description |
|--------|-------------|
| Forecast RMSE | Per-region temperature and precipitation error |
| Backward Transfer | Measures catastrophic forgetting across time periods |
| Regional Fairness | Variance in performance across geographic nodes |
| Communication Cost | Bytes exchanged per federated round |