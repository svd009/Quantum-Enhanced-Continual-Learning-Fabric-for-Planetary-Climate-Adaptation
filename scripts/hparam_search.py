"""
Hyperparameter search for federated climate training.
Runs a grid search over key hyperparameters and saves results.

Usage:
    python scripts/hparam_search.py --n-years 3 --output results/hparam_results.csv
"""
import sys
import os
import argparse
import itertools
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.utils.logging import get_logger
from src.data.synthetic import get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.evaluation.metrics import evaluate

logger = get_logger(__name__)

# ─── Search space ──────────────────────────────────────────────────────────────
SEARCH_SPACE = {
    "learning_rate": [0.01, 0.001, 0.0001],
    "ewc_lambda":    [1000, 5000, 10000],
    "pde_weight":    [0.01, 0.1, 0.5],
}


def make_cfg(lr: float, ewc_lambda: float, pde_weight: float, num_rounds: int, batch_size: int):
    return OmegaConf.create({
        "project": {"name": "fedclimate-hparam", "seed": 42, "device": "cpu"},
        "pinn": {
            "enabled": True,
            "pde_weight": pde_weight,
            "architecture": {
                "hidden_dims": [64, 64, 32],
                "activation": "tanh",
                "fourier_features": True,
                "fourier_scale": 10.0,
            },
            "physics": {"continuity_weight": 0.05},
        },
        "federated": {
            "enabled": True,
            "num_rounds": num_rounds,
            "num_clients": 3,
            "clients_per_round": 3,
            "local_epochs": 2,
            "aggregation": "fedavg",
        },
        "ewc": {"enabled": True, "lambda_": ewc_lambda, "fisher_samples": 50},
        "marl": {
            "enabled": False,
            "algorithm": "ppo",
            "num_agents": 3,
            "obs_dim": 8,
            "action_dim": 3,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "n_steps": 2048,
            "reward": {
                "forecast_accuracy_weight": 0.6,
                "fairness_weight": 0.2,
                "energy_weight": 0.2,
            },
        },
        "training": {
            "epochs": 100,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
        },
        "logging": {"use_wandb": False, "log_every": 10, "save_every": 25},
    })


def run_single(cfg, datasets, regions) -> dict:
    device = torch.device("cpu")
    client_loaders = []
    train_stats = None

    for region in regions:
        train_ds, _, _ = split_dataset_by_time(datasets[region])
        loader, stats = make_dataloader(train_ds, batch_size=cfg.training.batch_size)
        client_loaders.append(loader)
        if train_stats is None:
            train_stats = stats

    val_ds = split_dataset_by_time(datasets[regions[0]])[1]
    val_loader, _ = make_dataloader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, stats=train_stats
    )

    clients = [FederatedClient(r, cfg, device) for r in regions]
    server  = FederatedServer(cfg, device)
    history = server.run(
        clients=clients,
        client_dataloaders=client_loaders,
        val_dataloader=val_loader,
        eval_fn=evaluate,
    )

    final = history[-1]
    return {
        "final_rmse": final.get("rmse", 999),
        "final_mae":  final.get("mae",  999),
        "n_rounds":   len(history),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-years",    type=int, default=3)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output",     default="results/hparam_results.csv")
    parser.add_argument("--quick",      action="store_true",
                        help="Quick search with fewer combinations")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    regions = ["north_america", "europe", "asia_pacific"]
    logger.info(f"Loading synthetic data ({args.n_years} years)...")
    datasets = get_regional_datasets(regions=regions, n_years=args.n_years, seed=42)

    # Build search grid
    search_space = SEARCH_SPACE.copy()
    if args.quick:
        search_space = {
            "learning_rate": [0.01, 0.001],
            "ewc_lambda":    [1000, 5000],
            "pde_weight":    [0.01, 0.1],
        }

    keys   = list(search_space.keys())
    values = list(search_space.values())
    combos = list(itertools.product(*values))

    logger.info(f"Running {len(combos)} hyperparameter combinations...")
    logger.info(f"Search space: {search_space}")

    rows = []
    best_rmse = float("inf")
    best_params = {}

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        logger.info(f"\n[{i+1}/{len(combos)}] {params}")

        cfg = make_cfg(
            lr=params["learning_rate"],
            ewc_lambda=params["ewc_lambda"],
            pde_weight=params["pde_weight"],
            num_rounds=args.num_rounds,
            batch_size=args.batch_size,
        )

        try:
            results = run_single(cfg, datasets, regions)
            row = {**params, **results, "status": "success"}
            rows.append(row)

            if results["final_rmse"] < best_rmse:
                best_rmse   = results["final_rmse"]
                best_params = params.copy()
                best_params["rmse"] = best_rmse

            logger.info(f"  RMSE={results['final_rmse']:.4f} | MAE={results['final_mae']:.4f}")

        except Exception as e:
            logger.error(f"  Failed: {e}")
            rows.append({**params, "final_rmse": None, "final_mae": None, "status": str(e)})

    # Save results
    df = pd.DataFrame(rows)
    df = df.sort_values("final_rmse")
    df.to_csv(args.output, index=False)

    logger.info(f"\nResults saved to {args.output}")
    logger.info(f"\nTop 5 configurations:")
    logger.info("\n" + df.head(5).to_string(index=False))

    logger.info(f"\nBest hyperparameters:")
    for k, v in best_params.items():
        logger.info(f"  {k}: {v}")

    # Save best params as JSON
    best_path = "results/best_hparams.json"
    with open(best_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Best params saved to {best_path}")


if __name__ == "__main__":
    main()