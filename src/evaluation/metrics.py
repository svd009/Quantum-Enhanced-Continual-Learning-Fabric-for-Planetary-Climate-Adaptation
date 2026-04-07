"""
Evaluation metrics for climate forecasting.
These are the numbers that go in your ablation table.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Square Error — primary forecast accuracy metric."""
    return torch.sqrt(nn.functional.mse_loss(pred, target)).item()


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error — less sensitive to outliers than RMSE."""
    return nn.functional.l1_loss(pred, target).item()


def skill_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    climatology: torch.Tensor,
) -> float:
    """
    Murphy Skill Score vs climatology baseline.
    SS = 1 - MSE(model) / MSE(climatology)

    SS > 0 : model beats climatology
    SS = 0 : model equals climatology
    SS < 0 : model worse than climatology

    Climatology = simply predicting the mean of the training set.
    This is the standard benchmark in climate forecasting.
    """
    mse_model = nn.functional.mse_loss(pred, target).item()
    mse_clim = nn.functional.mse_loss(climatology, target).item()
    return 1.0 - mse_model / (mse_clim + 1e-8)


def regional_fairness(per_region_rmse: Dict[str, float]) -> float:
    """
    Fairness metric: how evenly does the model perform across regions?
    Score = 1 - (std / mean) of regional RMSEs.

    Score = 1.0 : perfect fairness (all regions equal)
    Score < 1.0 : some regions significantly worse than others

    This matters for climate equity — a model that works well for
    North America but poorly for Asia-Pacific is not truly global.
    """
    vals = list(per_region_rmse.values())
    mean = np.mean(vals)
    std = np.std(vals)
    return float(1.0 - std / (mean + 1e-8))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run full evaluation on a dataloader.
    Returns dict of metrics ready for logging and ablation table.
    """
    model.eval()
    all_pred = []
    all_target = []

    for coords, target in dataloader:
        coords = coords.to(device)
        pred = model(coords).cpu()
        all_pred.append(pred)
        all_target.append(target)

    pred_cat = torch.cat(all_pred)
    target_cat = torch.cat(all_target)

    # Climatology = mean of targets (simplest possible baseline)
    climatology = target_cat.mean(0, keepdim=True).expand_as(target_cat)

    return {
        "rmse":        rmse(pred_cat, target_cat),
        "mae":         mae(pred_cat, target_cat),
        "skill_score": skill_score(pred_cat, target_cat, climatology),
    }