"""
Main experiment entry point.
Run with: python scripts/run_experiment.py --config configs/experiment.yaml --synthetic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from omegaconf import OmegaConf

from src.utils.config import load_config, get_device, set_seed
from src.utils.logging import get_logger, ExperimentLogger
from src.data.synthetic import get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time
from src.models.pinn import ClimatePINN
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.evaluation.metrics import evaluate

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--n-years", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg)
    set_seed(cfg.project.seed)

    exp_logger = ExperimentLogger(cfg, run_name=args.run_name)
    logger.info(f"Device: {device}")
    logger.info(f"Synthetic mode: {args.synthetic}")

    # Load data
    regions = ["north_america", "europe", "asia_pacific"]
    logger.info("Generating synthetic climate data...")
    regional_datasets = get_regional_datasets(
        regions=regions,
        n_years=args.n_years,
        seed=cfg.project.seed,
    )

    # Build dataloaders per region
    client_loaders = []
    train_stats_list = []

    for region in regions:
        ds = regional_datasets[region]
        train_ds, val_ds, test_ds = split_dataset_by_time(ds)
        loader, stats = make_dataloader(
            train_ds,
            batch_size=cfg.training.batch_size,
            shuffle=True,
        )
        client_loaders.append(loader)
        train_stats_list.append(stats)

    # Use first region stats for val/test (shared normalization)
    val_ds = split_dataset_by_time(regional_datasets["north_america"])[1]
    val_loader, _ = make_dataloader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        stats=train_stats_list[0],
    )

    # Build clients
    clients = [
        FederatedClient(
            client_id=region,
            cfg=cfg,
            device=device,
        )
        for region in regions
    ]

    # Build and run server
    server = FederatedServer(cfg, device)
    history = server.run(
        clients=clients,
        client_dataloaders=client_loaders,
        val_dataloader=val_loader,
        eval_fn=evaluate,
    )

    # Log all rounds to W&B
    for round_metrics in history:
        step = round_metrics.get("round", 0)
        exp_logger.log(
            {k: v for k, v in round_metrics.items() if isinstance(v, float)},
            step=step,
        )

    # Log final summary
    final = history[-1]
    exp_logger.log_summary({
        "final_rmse": final.get("rmse", 0),
        "final_mae":  final.get("mae", 0),
    })

    # Log figures if they exist
    import os
    for fig_name, fig_path in [
        ("training_curves",     "results/figures/training_curves.png"),
        ("regional_comparison", "results/figures/regional_comparison.png"),
        ("ablation_table",      "results/figures/ablation_table.png"),
    ]:
        if os.path.exists(fig_path):
            exp_logger.log_figure(fig_name, fig_path)

    logger.info("Final RMSE: " + str(round(final.get("rmse", 0), 4)))
    logger.info("Final MAE:  " + str(round(final.get("mae", 0), 4)))

    exp_logger.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()