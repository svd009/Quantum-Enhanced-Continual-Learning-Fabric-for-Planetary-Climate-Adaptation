"""
Model checkpointing utilities.
Save and load model state during and after training.
"""
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from .logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during federated training.

    Features:
    - Save checkpoint every N rounds
    - Keep only the best K checkpoints (by RMSE)
    - Resume training from any checkpoint
    - Save final model with metadata

    Usage:
        ckpt = CheckpointManager(save_dir="results/checkpoints", keep_best=3)
        ckpt.save(model, round_idx=5, metrics={"rmse": 0.98})
        model = ckpt.load_best(model)
    """

    def __init__(
        self,
        save_dir: str = "results/checkpoints",
        keep_best: int = 3,
        save_every: int = 5,
    ):
        self.save_dir  = Path(save_dir)
        self.keep_best = keep_best
        self.save_every = save_every
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._registry: list = []
        self._registry_path = self.save_dir / "registry.json"

        if self._registry_path.exists():
            with open(self._registry_path) as f:
                self._registry = json.load(f)
            logger.info(f"Loaded checkpoint registry: {len(self._registry)} entries")

    def should_save(self, round_idx: int) -> bool:
        return (round_idx + 1) % self.save_every == 0

    def save(
        self,
        model: nn.Module,
        round_idx: int,
        metrics: Dict[str, Any],
        optimizer_state: Optional[Dict] = None,
        tag: str = "",
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ckpt_round{round_idx:04d}_{tag}_{timestamp}.pt"
        path = self.save_dir / name

        payload = {
            "round":           round_idx,
            "model_state":     model.state_dict(),
            "metrics":         metrics,
            "timestamp":       timestamp,
        }
        if optimizer_state is not None:
            payload["optimizer_state"] = optimizer_state

        torch.save(payload, path)

        entry = {
            "path":       str(path),
            "round":      round_idx,
            "rmse":       float(metrics.get("rmse", 999)),
            "timestamp":  timestamp,
            "tag":        tag,
        }
        self._registry.append(entry)
        self._save_registry()
        self._prune()

        logger.info(
            f"Checkpoint saved | round={round_idx} | "
            f"rmse={metrics.get('rmse', 'N/A'):.4f} | {path.name}"
        )
        return path

    def load(
        self,
        model: nn.Module,
        path: str,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location=device)
        model.load_state_dict(payload["model_state"])
        logger.info(
            f"Checkpoint loaded | round={payload['round']} | "
            f"rmse={payload['metrics'].get('rmse', 'N/A')}"
        )
        return payload

    def load_best(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, Any]]:
        if not self._registry:
            logger.warning("No checkpoints found in registry")
            return None

        best = min(self._registry, key=lambda x: x["rmse"])
        logger.info(
            f"Loading best checkpoint | round={best['round']} | rmse={best['rmse']:.4f}"
        )
        return self.load(model, best["path"], device)

    def load_latest(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> Optional[Dict[str, Any]]:
        if not self._registry:
            logger.warning("No checkpoints found")
            return None

        latest = max(self._registry, key=lambda x: x["round"])
        logger.info(
            f"Loading latest checkpoint | round={latest['round']} | rmse={latest['rmse']:.4f}"
        )
        return self.load(model, latest["path"], device)

    def save_final(
        self,
        model: nn.Module,
        history: list,
        metrics: Dict[str, Any],
    ) -> Path:
        path = self.save_dir / "final_model.pt"
        torch.save({
            "model_state": model.state_dict(),
            "history":     history,
            "metrics":     metrics,
            "timestamp":   datetime.now().strftime("%Y%m%d_%H%M%S"),
        }, path)
        logger.info(f"Final model saved: {path}")
        return path

    def list_checkpoints(self) -> list:
        return sorted(self._registry, key=lambda x: x["rmse"])

    def _save_registry(self) -> None:
        with open(self._registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def _prune(self) -> None:
        if len(self._registry) <= self.keep_best:
            return

        sorted_by_rmse = sorted(self._registry, key=lambda x: x["rmse"])
        to_keep  = set(e["path"] for e in sorted_by_rmse[:self.keep_best])
        to_delete = [e for e in self._registry if e["path"] not in to_keep]

        for entry in to_delete:
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
                logger.info(f"Pruned checkpoint: {p.name}")

        self._registry = [e for e in self._registry if e["path"] in to_keep]
        self._save_registry()