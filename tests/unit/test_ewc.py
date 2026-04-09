import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from omegaconf import OmegaConf
from src.models.pinn import ClimatePINN
from src.federated.ewc import EWC


@pytest.fixture
def model_and_loader():
    cfg = OmegaConf.create({
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
        }
    })
    model = ClimatePINN(cfg)
    coords = torch.randn(32, 3)
    targets = torch.randn(32, 5)
    loader = DataLoader(TensorDataset(coords, targets), batch_size=8)
    return model, loader


def test_penalty_zero_before_update(model_and_loader):
    model, _ = model_and_loader
    ewc = EWC(model, lam=100.0)
    penalty = ewc.penalty(model)
    assert penalty.item() == 0.0


def test_penalty_nonzero_after_update(model_and_loader):
    model, loader = model_and_loader
    ewc = EWC(model, lam=100.0)
    ewc.update(model, loader, torch.device("cpu"), n_samples=16)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    penalty = ewc.penalty(model)
    assert penalty.item() > 0.0


def test_penalty_zero_when_weights_unchanged(model_and_loader):
    model, loader = model_and_loader
    ewc = EWC(model, lam=100.0)
    ewc.update(model, loader, torch.device("cpu"), n_samples=16)
    penalty = ewc.penalty(model)
    assert penalty.item() < 1e-5


def test_fisher_keys_match_params(model_and_loader):
    model, loader = model_and_loader
    ewc = EWC(model, lam=100.0)
    ewc.update(model, loader, torch.device("cpu"), n_samples=16)
    param_names = {n for n, _ in model.named_parameters() if _.requires_grad}
    fisher_names = set(ewc._fisher.keys())
    assert param_names == fisher_names