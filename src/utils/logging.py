"""
Logging utilities with Weights & Biases integration.
Falls back to console logging if W&B is not configured.
"""
import logging
from typing import Optional, Dict, Any, List


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
    """
    Experiment logger with W&B integration.

    Usage:
        logger = ExperimentLogger(cfg, run_name="full_model_v1")
        logger.log({"rmse": 0.98, "loss": 0.45}, step=1)
        logger.log_summary({"best_rmse": 0.92})
        logger.finish()

    W&B setup:
        pip install wandb
        wandb login
        Set cfg.logging.use_wandb = true
    """

    def __init__(self, cfg, run_name: Optional[str] = None):
        self.cfg = cfg
        self._wandb = None
        self._run = None
        self.logger = get_logger("experiment")
        self._history: List[Dict] = []

        if cfg.logging.get("use_wandb", False):
            self._init_wandb(run_name)

    def _init_wandb(self, run_name: Optional[str]) -> None:
        try:
            import wandb

            config_dict = {}
            try:
                from omegaconf import OmegaConf
                config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            except Exception:
                config_dict = dict(self.cfg)

            self._run = wandb.init(
                project=self.cfg.project.name,
                name=run_name,
                config=config_dict,
                tags=[
                    "federated",
                    "continual-learning",
                    "climate",
                    "pinn",
                    "marl",
                ],
                notes="Federated PINN + EWC + MARL for climate forecasting",
            )
            self._wandb = wandb
            self.logger.info(f"W&B run initialized: {self._run.url}")

        except ImportError:
            self.logger.warning("wandb not installed. Run: pip install wandb")
        except Exception as e:
            self.logger.warning(f"W&B init failed: {e}. Logging to console only.")

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        self._history.append({"step": step, **metrics})
        msg = ", ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
            if isinstance(v, float)
        )
        self.logger.info(f"Step {step}: {msg}")
        if self._run is not None:
            self._wandb.log(metrics, step=step)

    def log_summary(self, metrics: Dict[str, Any]) -> None:
        self.logger.info("Summary: " + str(metrics))
        if self._run is not None:
            for k, v in metrics.items():
                self._wandb.run.summary[k] = v

    def log_ablation_table(self, rows: list, columns: list) -> None:
        if self._run is not None:
            table = self._wandb.Table(columns=columns, data=rows)
            self._wandb.log({"ablation_results": table})
            self.logger.info("Ablation table logged to W&B")

    def log_figure(self, name: str, path: str) -> None:
        if self._run is not None:
            self._wandb.log({name: self._wandb.Image(path)})
            self.logger.info(f"Figure logged to W&B: {name}")

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self.logger.info("W&B run finished")

    @property
    def history(self) -> List[Dict]:
        return self._history