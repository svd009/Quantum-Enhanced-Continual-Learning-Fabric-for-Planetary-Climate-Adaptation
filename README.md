# Quantum-Enhanced Continual Learning Fabric for Planetary Climate Adaptation

A production-grade federated learning system for planetary climate forecasting, built for applied scientist roles at top-tier ML companies.

## What This Project Does

Trains climate forecasting models across geographically distributed nodes (North America, Europe, Asia-Pacific) without centralizing raw data, while preventing catastrophic forgetting as climate distributions shift over time.

## Architecture

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Federated Learning | FedAvg across 3 regional nodes | Train without centralizing data |
| Continual Learning | Elastic Weight Consolidation (EWC) | Prevent forgetting across time periods |
| Physics-Informed ML | PINN with PDE residual loss | Enforce mass + energy conservation |
| Multi-Agent RL | PPO agents per region | Adaptive regional coordination |

## Ablation Matrix

| Condition | Federated | EWC | PINN | MARL |
|-----------|-----------|-----|------|------|
| Centralized baseline | — | — | — | — |
| + Federated | ✓ | — | — | — |
| + EWC | ✓ | ✓ | — | — |
| + PINN | ✓ | ✓ | ✓ | — |
| Full model | ✓ | ✓ | ✓ | ✓ |

## Project Structure
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

## Quickstart
```bash
# Install
pip install -e .

# Run with synthetic data (no API key needed)
python scripts/run_experiment.py --config configs/experiment.yaml --synthetic

# Run full ablation suite
python scripts/run_ablation.py --output results/ablation_table.csv
```

## Data Sources

- [ERA5 Reanalysis](https://cds.climate.copernicus.eu/) — hourly climate variables 1940–present
- [CMIP6 Projections](https://esgf-node.llnl.gov/projects/cmip6/) — future climate scenarios
- Synthetic generator included for local development (no API key required)

## Tech Stack

- **PyTorch 2.0+** — model training
- **xarray** — climate data handling
- **OmegaConf** — hierarchical config management
- **AWS SageMaker** — multi-node federated runs (Week 4+)

## Key Metrics

- Forecast RMSE per region
- Backward Transfer (forgetting metric)
- Regional Fairness Score
- Communication cost per federated round