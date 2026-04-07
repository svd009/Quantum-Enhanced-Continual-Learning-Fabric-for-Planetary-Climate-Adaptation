"""
Ablation runner — trains and evaluates all conditions,
produces the comparison table that goes in your portfolio.
"""
import pandas as pd
import torch
from typing import List, Dict, Callable, Any
from omegaconf import OmegaConf, DictConfig


ABLATION_CONDITIONS = [
    {
        "name": "centralized",
        "label": "Centralized baseline",
        "federated": False,
        "ewc": False,
        "pinn": False,
        "marl": False,
    },
    {
        "name": "federated_only",
        "label": "+ Federated",
        "federated": True,
        "ewc": False,
        "pinn": False,
        "marl": False,
    },
    {
        "name": "federated_ewc",
        "label": "+ EWC",
        "federated": True,
        "ewc": True,
        "pinn": False,
        "marl": False,
    },
    {
        "name": "federated_ewc_pinn",
        "label": "+ PINN",
        "federated": True,
        "ewc": True,
        "pinn": True,
        "marl": False,
    },
    {
        "name": "full_model",
        "label": "Full model",
        "federated": True,
        "ewc": True,
        "pinn": True,
        "marl": True,
    },
]


def run_ablation(
    base_cfg: DictConfig,
    train_fn: Callable,
    eval_fn: Callable,
    output_path: str = "results/ablation_table.csv",
) -> pd.DataFrame:
    """
    Run all ablation conditions and return a results DataFrame.

    Args:
        base_cfg:    base OmegaConf config
        train_fn:    function(cfg) -> trained model
        eval_fn:     function(cfg, model) -> metrics dict
        output_path: where to save the CSV

    Returns:
        DataFrame with one row per ablation condition
    """
    rows = []

    for condition in ABLATION_CONDITIONS:
        print(f"\n{'='*50}")
        print(f"Running: {condition['label']}")
        print(f"{'='*50}")

        # Build condition-specific config
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create({
            "federated": {"enabled": condition["federated"]},
            "ewc":       {"enabled": condition["ewc"]},
            "pinn":      {"enabled": condition["pinn"]},
            "marl":      {"enabled": condition["marl"]},
        }))

        # Train and evaluate
        model = train_fn(cfg)
        metrics = eval_fn(cfg, model)

        rows.append({
            "Condition":  condition["label"],
            "Federated":  "✓" if condition["federated"] else "—",
            "EWC":        "✓" if condition["ewc"] else "—",
            "PINN":       "✓" if condition["pinn"] else "—",
            "MARL":       "✓" if condition["marl"] else "—",
            **metrics,
        })

        print(f"Results: {metrics}")

    df = pd.DataFrame(rows)

    # Save to CSV
    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nAblation table saved to {output_path}")
    print(df.to_string(index=False))

    return df