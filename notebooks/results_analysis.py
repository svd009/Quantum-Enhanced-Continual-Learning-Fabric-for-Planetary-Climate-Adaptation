"""
Results analysis and visualization.
Run with: python notebooks/results_analysis.py
Generates charts saved to results/figures/
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from omegaconf import OmegaConf

from src.data.synthetic import get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time
from src.models.pinn import ClimatePINN
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.evaluation.metrics import evaluate, regional_fairness
from src.evaluation.forgetting import backward_transfer

os.makedirs("results/figures", exist_ok=True)

REGIONS = ["north_america", "europe", "asia_pacific"]
REGION_LABELS = ["North America", "Europe", "Asia-Pacific"]

def get_cfg(federated=True, ewc=True, pinn=True):
    return OmegaConf.create({
        "project": {"name": "fedclimate", "seed": 42, "device": "cpu"},
        "pinn": {
            "enabled": pinn,
            "pde_weight": 0.1,
            "architecture": {
                "hidden_dims": [64, 64, 32],
                "activation": "tanh",
                "fourier_features": True,
                "fourier_scale": 10.0,
            },
            "physics": {"continuity_weight": 0.05},
        },
        "federated": {
            "enabled": federated,
            "num_rounds": 8,
            "num_clients": 3,
            "clients_per_round": 3,
            "local_epochs": 2,
            "aggregation": "fedavg",
        },
        "ewc": {"enabled": ewc, "lambda_": 5000, "fisher_samples": 50},
        "training": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
        },
        "logging": {"use_wandb": False, "log_every": 1, "save_every": 1},
    })


def run_condition(cfg, datasets, label):
    device = torch.device("cpu")
    client_loaders = []
    val_loaders = []

    for region in REGIONS:
        train_ds, val_ds, _ = split_dataset_by_time(datasets[region])
        loader, stats = make_dataloader(train_ds, batch_size=cfg.training.batch_size)
        val_loader, _ = make_dataloader(val_ds, batch_size=32, shuffle=False, stats=stats)
        client_loaders.append(loader)
        val_loaders.append(val_loader)

    clients = [FederatedClient(r, cfg, device) for r in REGIONS]
    server = FederatedServer(cfg, device)
    history = server.run(
        clients=clients,
        client_dataloaders=client_loaders,
        val_dataloader=val_loaders[0],
        eval_fn=evaluate,
    )

    # Per-region evaluation
    per_region_rmse = {}
    for i, region in enumerate(REGIONS):
        metrics = evaluate(server.global_model, val_loaders[i], device)
        per_region_rmse[region] = metrics["rmse"]

    fairness = regional_fairness(per_region_rmse)
    final_rmse = history[-1].get("rmse", 0)

    print(f"\n{label}:")
    print(f"  Final RMSE:   {final_rmse:.4f}")
    print(f"  Fairness:     {fairness:.4f}")
    for r, v in per_region_rmse.items():
        print(f"  {r}: {v:.4f}")

    return history, per_region_rmse, fairness


def plot_training_curves(all_histories, labels, colors):
    fig, ax = plt.subplots(figsize=(10, 5))
    for history, label, color in zip(all_histories, labels, colors):
        rounds = [h["round"] for h in history]
        rmses = [h["rmse"] for h in history]
        ax.plot(rounds, rmses, marker="o", label=label, color=color, linewidth=2)
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Validation RMSE", fontsize=12)
    ax.set_title("Training Convergence Across Ablation Conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/training_curves.png", dpi=150)
    plt.close()
    print("Saved: results/figures/training_curves.png")


def plot_regional_comparison(all_region_rmses, labels, colors):
    x = np.arange(len(REGION_LABELS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (rmses, label, color) in enumerate(zip(all_region_rmses, labels, colors)):
        vals = [rmses[r] for r in REGIONS]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(REGION_LABELS, fontsize=11)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Per-Region RMSE Across Ablation Conditions", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/figures/regional_comparison.png", dpi=150)
    plt.close()
    print("Saved: results/figures/regional_comparison.png")


def plot_ablation_table(results):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    columns = ["Condition", "Federated", "EWC", "PINN", "RMSE", "Fairness"]
    rows = []
    for r in results:
        rows.append([
            r["label"],
            "✓" if r["federated"] else "—",
            "✓" if r["ewc"] else "—",
            "✓" if r["pinn"] else "—",
            f"{r['rmse']:.4f}",
            f"{r['fairness']:.4f}",
        ])
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F3F4")
    plt.title("Ablation Study Results", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("results/figures/ablation_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/figures/ablation_table.png")


def main():
    print("Loading synthetic climate data...")
    datasets = get_regional_datasets(REGIONS, n_years=5, seed=42)

    conditions = [
        {"label": "Federated only", "federated": True,  "ewc": False, "pinn": False, "color": "#3498DB"},
        {"label": "Fed + EWC",      "federated": True,  "ewc": True,  "pinn": False, "color": "#E67E22"},
        {"label": "Full model",     "federated": True,  "ewc": True,  "pinn": True,  "color": "#27AE60"},
    ]

    all_histories = []
    all_region_rmses = []
    results_table = []

    for cond in conditions:
        cfg = get_cfg(
            federated=cond["federated"],
            ewc=cond["ewc"],
            pinn=cond["pinn"],
        )
        history, region_rmses, fairness = run_condition(cfg, datasets, cond["label"])
        all_histories.append(history)
        all_region_rmses.append(region_rmses)
        results_table.append({
            "label":     cond["label"],
            "federated": cond["federated"],
            "ewc":       cond["ewc"],
            "pinn":      cond["pinn"],
            "rmse":      history[-1]["rmse"],
            "fairness":  fairness,
        })

    print("\nGenerating charts...")
    plot_training_curves(
        all_histories,
        [c["label"] for c in conditions],
        [c["color"] for c in conditions],
    )
    plot_regional_comparison(
        all_region_rmses,
        [c["label"] for c in conditions],
        [c["color"] for c in conditions],
    )
    plot_ablation_table(results_table)

    print("\nAll figures saved to results/figures/")
    print("Done.")


if __name__ == "__main__":
    main()