"""
Reward functions for MARL climate adaptation agents.
Each agent (region) receives a reward based on:
1. Forecast accuracy (primary signal)
2. Fairness across regions (penalize inequality)
3. Compute efficiency (penalize expensive updates)
"""
import torch
import numpy as np
from typing import Dict, List


def forecast_accuracy_reward(rmse: float, baseline_rmse: float) -> float:
    """
    Reward for beating the climatology baseline.
    Positive when model beats baseline, negative when worse.
    """
    return float(baseline_rmse - rmse) / (baseline_rmse + 1e-8)


def fairness_penalty(regional_rmses: Dict[str, float]) -> float:
    """
    Penalize high variance in performance across regions.
    All regions should improve together.
    """
    vals = list(regional_rmses.values())
    if len(vals) < 2:
        return 0.0
    return float(-np.std(vals) / (np.mean(vals) + 1e-8))


def compute_reward(
    agent_id: str,
    regional_rmses: Dict[str, float],
    baseline_rmse: float,
    weights: Dict[str, float] = None,
) -> float:
    """
    Compute total reward for one agent in one step.

    Args:
        agent_id:       which region this agent represents
        regional_rmses: current RMSE per region
        baseline_rmse:  climatology baseline RMSE
        weights:        reward component weights

    Returns:
        scalar reward
    """
    if weights is None:
        weights = {
            "accuracy": 0.6,
            "fairness": 0.2,
            "efficiency": 0.2,
        }

    agent_rmse = regional_rmses.get(agent_id, baseline_rmse)

    r_accuracy = forecast_accuracy_reward(agent_rmse, baseline_rmse)
    r_fairness = fairness_penalty(regional_rmses)
    r_efficiency = -0.01

    total = (
        weights["accuracy"] * r_accuracy
        + weights["fairness"] * r_fairness
        + weights["efficiency"] * r_efficiency
    )

    return float(total)