from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import torch
import random
import numpy as np


def load_config(*config_paths: str) -> DictConfig:
    configs = [OmegaConf.load(p) for p in config_paths]
    cfg = OmegaConf.merge(*configs)
    return cfg


def get_device(cfg: DictConfig) -> torch.device:
    requested = cfg.project.get("device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)