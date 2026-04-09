import pytest
import torch
from omegaconf import OmegaConf
from src.models.pinn import ClimatePINN


@pytest.fixture
def cfg():
    return OmegaConf.create({
        "pinn": {
            "enabled": True,
            "pde_weight": 0.1,
            "architecture": {
                "hidden_dims": [32, 32],
                "activation": "tanh",
                "fourier_features": True,
                "fourier_scale": 10.0,
            },
            "physics": {"continuity_weight": 0.05},
        }
    })


def test_forward_shape(cfg):
    model = ClimatePINN(cfg)
    coords = torch.randn(16, 3)
    out = model(coords)
    assert out.shape == (16, 5)


def test_loss_keys(cfg):
    model = ClimatePINN(cfg)
    coords = torch.randn(8, 3)
    target = torch.randn(8, 5)
    loss_dict = model.compute_loss(coords, target)
    assert "total" in loss_dict
    assert "data" in loss_dict


def test_loss_is_differentiable(cfg):
    model = ClimatePINN(cfg)
    coords = torch.randn(8, 3)
    target = torch.randn(8, 5)
    loss_dict = model.compute_loss(coords, target)
    assert loss_dict["total"].requires_grad


def test_pinn_disabled(cfg):
    cfg.pinn.enabled = False
    model = ClimatePINN(cfg)
    coords = torch.randn(8, 3)
    target = torch.randn(8, 5)
    loss_dict = model.compute_loss(coords, target)
    assert "total" in loss_dict


def test_predict_no_grad(cfg):
    model = ClimatePINN(cfg)
    coords = torch.randn(8, 3)
    out = model.predict(coords, torch.device("cpu"))
    assert out.shape == (8, 5)