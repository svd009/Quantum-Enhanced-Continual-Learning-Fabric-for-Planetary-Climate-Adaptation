import pytest
import torch
from omegaconf import OmegaConf
from src.data.synthetic import get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import fedavg
from src.evaluation.metrics import evaluate


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "project": {"name": "test", "seed": 42, "device": "cpu"},
        "pinn": {
            "enabled": False,
            "pde_weight": 0.0,
            "architecture": {
                "hidden_dims": [32, 32],
                "activation": "tanh",
                "fourier_features": False,
                "fourier_scale": 1.0,
            },
            "physics": {"continuity_weight": 0.0},
        },
        "federated": {
            "enabled": True,
            "num_rounds": 2,
            "num_clients": 2,
            "clients_per_round": 2,
            "local_epochs": 1,
            "aggregation": "fedavg",
        },
        "ewc": {"enabled": False, "lambda_": 0, "fisher_samples": 10},
        "training": {
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
        },
        "logging": {"use_wandb": False, "log_every": 1, "save_every": 1},
    })


def test_fedavg_aggregation(cfg):
    device = torch.device("cpu")
    clients = [
        FederatedClient("na", cfg, device),
        FederatedClient("eu", cfg, device),
    ]
    states = [c.get_parameters() for c in clients]
    weights = [0.5, 0.5]
    server = FederatedServer(cfg, device)
    fedavg(server.global_model, states, weights)
    assert server.global_model is not None


def test_single_federated_round(cfg):
    device = torch.device("cpu")
    regions = ["north_america", "europe"]
    datasets = get_regional_datasets(regions, n_years=2)

    loaders = []
    for r in regions:
        train_ds, _, _ = split_dataset_by_time(datasets[r])
        loader, _ = make_dataloader(train_ds, batch_size=16)
        loaders.append(loader)

    val_ds = split_dataset_by_time(datasets["north_america"])[1]
    val_loader, _ = make_dataloader(val_ds, batch_size=16, shuffle=False)

    clients = [FederatedClient(r, cfg, device) for r in regions]
    server = FederatedServer(cfg, device)

    history = server.run(
        clients=clients,
        client_dataloaders=loaders,
        val_dataloader=val_loader,
        eval_fn=evaluate,
    )

    assert len(history) == 2
    assert "rmse" in history[-1]
    assert history[-1]["rmse"] > 0