import pytest
import torch
from src.evaluation.metrics import rmse, mae, skill_score, regional_fairness


def test_rmse_perfect():
    x = torch.randn(100, 5)
    assert rmse(x, x) < 1e-5


def test_rmse_known_value():
    pred = torch.zeros(4, 1)
    target = torch.ones(4, 1)
    assert abs(rmse(pred, target) - 1.0) < 1e-5


def test_mae_perfect():
    x = torch.randn(50, 5)
    assert mae(x, x) < 1e-5


def test_skill_score_perfect():
    target = torch.randn(50, 5)
    climatology = torch.zeros_like(target)
    ss = skill_score(target, target, climatology)
    assert ss == pytest.approx(1.0, abs=1e-5)


def test_skill_score_climatology():
    target = torch.randn(50, 5)
    climatology = torch.zeros_like(target)
    ss = skill_score(climatology, target, climatology)
    assert ss == pytest.approx(0.0, abs=1e-5)


def test_regional_fairness_perfect():
    scores = {"north_america": 0.5, "europe": 0.5, "asia_pacific": 0.5}
    assert regional_fairness(scores) == pytest.approx(1.0, abs=1e-5)


def test_regional_fairness_unequal():
    scores = {"north_america": 0.2, "europe": 0.8, "asia_pacific": 0.5}
    assert regional_fairness(scores) < 1.0