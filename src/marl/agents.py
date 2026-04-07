"""
PPO agent for climate adaptation.
One agent per geographic region.
Uses a simple actor-critic architecture.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class ActorCritic(nn.Module):
    """
    Shared backbone with separate actor and critic heads.
    Actor:  outputs action mean for continuous action space
    Critic: outputs state value estimate
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)
        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp()
        value = self.critic(features)
        return action_mean, action_std, value

    def get_action(self, obs: torch.Tensor):
        action_mean, action_std, value = self.forward(obs)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor):
        action_mean, action_std, value = self.forward(obs)
        dist = torch.distributions.Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """
    PPO agent for one climate region.

    Actions control adaptation parameters:
        action[0]: learning rate multiplier
        action[1]: PDE loss weight
        action[2]: local training epochs
    """

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 8,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_range: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_id = agent_id
        self.gamma = gamma
        self.clip_range = clip_range
        self.device = device

        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.obs_buffer: List = []
        self.action_buffer: List = []
        self.reward_buffer: List = []
        self.log_prob_buffer: List = []
        self.value_buffer: List = []

    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(obs_tensor)
        self.obs_buffer.append(obs)
        self.action_buffer.append(action.cpu().numpy()[0])
        self.log_prob_buffer.append(log_prob.cpu().item())
        self.value_buffer.append(value.cpu().item())
        return action.cpu().numpy()[0]

    def store_reward(self, reward: float):
        self.reward_buffer.append(reward)

    def update(self) -> Dict[str, float]:
        if len(self.reward_buffer) < 2:
            return {}

        returns = []
        G = 0.0
        for r in reversed(self.reward_buffer):
            G = r + self.gamma * G
            returns.insert(0, G)

        obs = torch.FloatTensor(np.array(self.obs_buffer)).to(self.device)
        actions = torch.FloatTensor(np.array(self.action_buffer)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_prob_buffer).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.value_buffer).to(self.device)

        advantages = returns_t - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs, new_values, entropy = self.policy.evaluate_action(obs, actions)
        ratio = (log_probs - old_log_probs).exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_range, 1 + self.clip_range) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.functional.mse_loss(new_values, returns_t)
        entropy_loss = -entropy.mean()

        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.log_prob_buffer.clear()
        self.value_buffer.clear()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": (-entropy_loss).item(),
        }