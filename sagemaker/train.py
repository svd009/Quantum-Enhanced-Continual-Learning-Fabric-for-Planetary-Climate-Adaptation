import os
import sys
import json
import torch
import argparse

SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "model")
SM_NUM_GPUS = int(os.environ.get("SM_NUM_GPUS", 0))

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.utils.logging import get_logger
from src.data.synthetic import get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.evaluation.metrics import evaluate

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-years", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--ewc-lambda", type=float, default=5000.0)
    parser.add_argument("--pde-weight", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("SageMaker training job starting")
    logger.info("GPUs available: " + str(SM_NUM_GPUS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: " + str(device))

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "project": {"name": "fedclimate-sagemaker", "seed": 42, "device": str(device)},
        "pinn": {
            "enabled": True,
            "pde_weight": args.pde_weight,
            "architecture": {
                "hidden_dims": [256, 256, 128, 64],
                "activation": "tanh",
                "fourier_features": True,
                "fourier_scale": 10.0,
            },
            "physics": {"continuity_weight": 0.05},
        },
        "federated": {
            "enabled": True,
            "num_rounds": args.num_rounds,
            "num_clients": 3,
            "clients_per_round": 3,
            "local_epochs": 5,
            "aggregation": "fedavg",
        },
        "ewc": {"enabled": True, "lambda_": args.ewc_lambda, "fisher_samples": 200},
        "marl": {
            "enabled": True,
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
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
        },
        "logging": {"use_wandb": False, "log_every": 5, "save_every": 10},
    })

    regions = ["north_america", "europe", "asia_pacific"]
    datasets = get_regional_datasets(regions=regions, n_years=args.n_years, seed=42)

    client_loaders = []
    train_stats = None
    for region in regions:
        train_ds, _, _ = split_dataset_by_time(datasets[region])
        loader, stats = make_dataloader(train_ds, batch_size=args.batch_size)
        client_loaders.append(loader)
        if train_stats is None:
            train_stats = stats

    val_ds = split_dataset_by_time(datasets["north_america"])[1]
    val_loader, _ = make_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, stats=train_stats
    )

    clients = [FederatedClient(r, cfg, device) for r in regions]
    server = FederatedServer(cfg, device)
    history = server.run(
        clients=clients,
        client_dataloaders=client_loaders,
        val_dataloader=val_loader,
        eval_fn=evaluate,
    )

    final = history[-1]
    logger.info("Final RMSE: " + str(round(final.get("rmse", 0), 4)))

    os.makedirs(SM_MODEL_DIR, exist_ok=True)
    model_path = os.path.join(SM_MODEL_DIR, "model.pth")
    torch.save({
        "model_state_dict": server.global_model.state_dict(),
        "history": history,
        "final_rmse": final.get("rmse", None),
    }, model_path)
    logger.info("Model saved to " + model_path)

    metrics_path = os.path.join(SM_MODEL_DIR, "metrics.json")
    with open(metrics_path, "w") as mf:
        json.dump({k: str(v) for k, v in final.items()}, mf, indent=2)
    logger.info("Metrics saved to " + metrics_path)


if __name__ == "__main__":
    main()