import pytest
import numpy as np
from src.marl.agents import PPOAgent
from src.marl.rewards import compute_reward, forecast_accuracy_reward, fairness_penalty


def test_agent_action_shape():
    agent = PPOAgent("north_america", obs_dim=8, action_dim=3)
    obs = np.random.randn(8).astype("float32")
    action = agent.select_action(obs)
    assert action.shape == (3,)


def test_agent_update_returns_metrics():
    agent = PPOAgent("north_america", obs_dim=8, action_dim=3)
    for _ in range(5):
        obs = np.random.randn(8).astype("float32")
        agent.select_action(obs)
        agent.store_reward(float(np.random.randn()))
    metrics = agent.update()
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics


def test_forecast_accuracy_reward_positive():
    reward = forecast_accuracy_reward(rmse=0.5, baseline_rmse=1.0)
    assert reward > 0.0


def test_forecast_accuracy_reward_negative():
    reward = forecast_accuracy_reward(rmse=1.5, baseline_rmse=1.0)
    assert reward < 0.0


def test_fairness_penalty_equal_regions():
    scores = {"na": 0.5, "eu": 0.5, "ap": 0.5}
    penalty = fairness_penalty(scores)
    assert penalty == pytest.approx(0.0, abs=1e-5)


def test_compute_reward_returns_float():
    reward = compute_reward(
        "north_america",
        {"north_america": 0.8, "europe": 0.9, "asia_pacific": 0.85},
        baseline_rmse=1.0,
    )
    assert isinstance(reward, float)