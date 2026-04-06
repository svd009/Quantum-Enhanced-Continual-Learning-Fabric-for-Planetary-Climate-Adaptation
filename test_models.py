from omegaconf import OmegaConf
from src.models.pinn import ClimatePINN
import torch

cfg = OmegaConf.create({
    "pinn": {
        "enabled": True,
        "pde_weight": 0.1,
        "architecture": {
            "hidden_dims": [64, 64],
            "activation": "tanh",
            "fourier_features": True,
            "fourier_scale": 10.0,
        },
        "physics": {"continuity_weight": 0.05},
    }
})

model = ClimatePINN(cfg)
coords = torch.randn(16, 3)
target = torch.randn(16, 5)
loss_dict = model.compute_loss(coords, target)
print(list(loss_dict.keys()))
print("models OK")