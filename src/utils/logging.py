import logging
from typing import Optional, Dict, Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class ExperimentLogger:
    def __init__(self, cfg, run_name: Optional[str] = None):
        self.cfg = cfg
        self._wandb = None
        self.logger = get_logger("experiment")

        if cfg.logging.get("use_wandb", False):
            try:
                import wandb
                self._wandb = wandb.init(
                    project=cfg.project.name,
                    name=run_name,
                    config=dict(cfg),
                )
            except Exception as e:
                self.logger.warning(f"W&B init failed: {e}. Logging to console only.")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        msg = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        self.logger.info(f"Step {step}: {msg}")
        if self._wandb:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._wandb:
            self._wandb.finish()