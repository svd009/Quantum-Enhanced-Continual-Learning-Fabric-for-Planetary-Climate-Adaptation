"""
Multi-agent climate adaptation environment.
Each agent controls one geographic region and decides
how to weight its local training signal each round.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from ..models.pinn import ClimatePINN
from ..evaluation.metrics import evaluate
from .rewards import compute_reward


class ClimateEnv:
    """
    Custom multi-agent environment for climate adaptation.

    State (per agent):
        - Current regional RMSE
        - Round number (normalized)
        - Loss trend (last 3 rounds)
        - Global model performance

    Action (per agent):
        - Learning rate multiplier [0.5, 2.0]
        - PDE weight multiplier [0.0, 2.0]
        - Local epochs [1, 10]

    Reward:
        - Forecast accuracy improvement
        - Regional fairness
        - Compute efficiency
    """

    def __init__(
        self,
        cfg,
        device: torch.device,
        client_dataloaders: List[DataLoader],
        val_dataloader: DataLoader,
        region_names: List[str],
    ):
        self.cfg = cfg
        self.device = device
        self.client_dataloaders = client_dataloaders
        self.val_dataloader = val_dataloader
        self.region_names = region_names
        self.n_agents = len(region_names)

        self.obs_dim = 8
        self.action_dim = 3

        self.reset()

    def reset(self) -> Dict[str, np.ndarray]:
        self.round = 0
        self.regional_rmses = {r: 1.0 for r in self.region_names}
        self.loss_history = {r: [1.0, 1.0, 1.0] for r in self.region_names}
        self.baseline_rmse = 1.0
        return self._get_obs()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = {}
        for region in self.region_names:
            rmse = self.regional_rmses[region]
            trend = self.loss_history[region]
            obs[region] = np.array([
                rmse,
                self.round / max(self.cfg.federated.num_rounds, 1),
                trend[-1] - trend[-2],
                trend[-2] - trend[-3],
                np.mean(list(self.regional_rmses.values())),
                np.std(list(self.regional_rmses.values())),
                self.baseline_rmse,
                rmse / (self.baseline_rmse + 1e-8),
            ], dtype=np.float32)
        return obs

    def step(
        self,
        actions: Dict[str, np.ndarray],
        models: Dict[str, ClimatePINN],
    ) -> Tuple[Dict, Dict, bool, Dict]:
        self.round += 1

        new_rmses = {}
        for region, model in models.items():
            metrics = evaluate(model, self.val_dataloader, self.device)
            new_rmses[region] = metrics["rmse"]
            self.loss_history[region].append(metrics["rmse"])
            self.loss_history[region] = self.loss_history[region][-3:]

        self.regional_rmses = new_rmses

        rewards = {
            region: compute_reward(region, self.regional_rmses, self.baseline_rmse)
            for region in self.region_names
        }

        obs = self._get_obs()
        done = self.round >= self.cfg.federated.num_rounds
        info = {"round": self.round, "regional_rmses": self.regional_rmses}

        return obs, rewards, done, info
