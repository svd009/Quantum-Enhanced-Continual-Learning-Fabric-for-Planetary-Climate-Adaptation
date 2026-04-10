"""
Federated learning server with optional MARL integration.
PPO agents per region adaptively control training parameters.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Callable, Optional
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ..models.pinn import ClimatePINN
from .client import FederatedClient
from .aggregation import fedavg


class FederatedServer:

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.global_model = ClimatePINN(cfg).to(device)
        self.history: List[Dict[str, Any]] = []
        self.marl_enabled = cfg.get("marl", {}).get("enabled", False)
        self.agents = {}
        self.agent_obs = {}

        if self.marl_enabled:
            from ..marl.agents import PPOAgent
            from ..marl.rewards import compute_reward
            self.PPOAgent = PPOAgent
            self.compute_reward = compute_reward
            regions = ["north_america", "europe", "asia_pacific"]
            for region in regions:
                self.agents[region] = PPOAgent(
                    agent_id=region,
                    obs_dim=8,
                    action_dim=3,
                    device=device,
                )
            print(f"  MARL enabled | agents={len(self.agents)}")

    def _get_agent_obs(self, region: str, round_idx: int, regional_rmses: Dict) -> Any:
        import numpy as np
        rmse = regional_rmses.get(region, 1.0)
        mean_rmse = sum(regional_rmses.values()) / max(len(regional_rmses), 1)
        std_rmse = (sum((v - mean_rmse) ** 2 for v in regional_rmses.values()) / max(len(regional_rmses), 1)) ** 0.5
        return {
            "rmse": rmse,
            "round_norm": round_idx / max(self.cfg.federated.num_rounds, 1),
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
        }

    def _apply_action(self, client: FederatedClient, action) -> None:
        import numpy as np
        lr_mult = float(np.clip(action[0], 0.5, 2.0))
        base_lr = self.cfg.training.learning_rate
        new_lr = base_lr * lr_mult
        if client.optimizer is not None:
            for pg in client.optimizer.param_groups:
                pg["lr"] = new_lr

    def run(
        self,
        clients: List[FederatedClient],
        client_dataloaders: List[DataLoader],
        val_dataloader: DataLoader,
        eval_fn: Callable,
    ) -> List[Dict[str, Any]]:
        import numpy as np
        n_rounds = self.cfg.federated.num_rounds
        print(f"\nFederated training | rounds={n_rounds} | clients={len(clients)} | marl={self.marl_enabled}\n")

        regional_rmses = {c.client_id: 1.0 for c in clients}
        baseline_rmse = 1.0

        for round_idx in range(n_rounds):
            print(f"--- Round {round_idx + 1}/{n_rounds} ---")

            # MARL: agents observe and act
            if self.marl_enabled:
                for client in clients:
                    obs_dict = self._get_agent_obs(client.client_id, round_idx, regional_rmses)
                    obs_arr = np.array(list(obs_dict.values()) + [0.0, 0.0, 0.0, 0.0], dtype="float32")[:8]
                    agent = self.agents[client.client_id]
                    action = agent.select_action(obs_arr)
                    self._apply_action(client, action)

            # Broadcast global model
            global_state = self.global_model.state_dict()
            for client in clients:
                client.set_parameters(global_state)

            # Local training
            client_states = []
            client_weights = []
            round_metrics = {"round": round_idx + 1}

            for client, dataloader in zip(clients, client_dataloaders):
                metrics = client.train(dataloader)
                client_states.append(client.get_parameters())
                client_weights.append(1.0 / len(clients))
                round_metrics[f"{client.client_id}_loss"] = metrics["train_loss"]

            # Aggregate
            fedavg(self.global_model, client_states, client_weights)

            # Update EWC
            for client, dataloader in zip(clients, client_dataloaders):
                client.update_ewc(dataloader)

            # Evaluate
            val_metrics = eval_fn(self.global_model, val_dataloader, self.device)
            round_metrics.update(val_metrics)
            self.history.append(round_metrics)

            # Update regional RMSEs
            for client in clients:
                regional_rmses[client.client_id] = val_metrics.get("rmse", 1.0)

            # MARL: store rewards and update policies
            if self.marl_enabled:
                for client in clients:
                    reward = self.compute_reward(
                        client.client_id,
                        regional_rmses,
                        baseline_rmse,
                    )
                    self.agents[client.client_id].store_reward(reward)

                # Update all agents every 2 rounds
                if (round_idx + 1) % 2 == 0:
                    for region, agent in self.agents.items():
                        marl_metrics = agent.update()
                        if marl_metrics:
                            round_metrics[f"{region}_actor_loss"] = marl_metrics.get("actor_loss", 0)

            print(f"  Val RMSE: {val_metrics.get('rmse', 0):.4f}\n")

        print("Training complete.")
        return self.history