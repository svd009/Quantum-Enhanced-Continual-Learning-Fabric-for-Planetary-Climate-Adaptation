"""
Normalize, split, and tensorize climate data.
Converts xarray Datasets into PyTorch DataLoaders.
"""
import torch
import numpy as np
import xarray as xr
from typing import Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader


VARIABLE_NAMES = ["t2m", "tp", "sp", "u10", "v10"]


class ClimateDataset(Dataset):
    """
    Converts xarray ERA5 (or synthetic) data into (coords, target) tensor pairs.

    coords : (lat_norm, lon_norm, time_norm) — shape (N, 3)
             all normalized to [0, 1] for PINN stability
    target : (t2m, tp, sp, u10, v10)         — shape (N, 5)
             z-score normalized using training set stats
    """

    def __init__(
        self,
        ds: xr.Dataset,
        stats: Optional[Dict] = None,
        variables: list = VARIABLE_NAMES,
    ):
        # Stack variables into (T, H, W, C)
        arrays = []
        for v in variables:
            if v in ds:
                arrays.append(ds[v].values)
            else:
                raise KeyError(f"Variable '{v}' not found in dataset. Available: {list(ds.data_vars)}")

        data = np.stack(arrays, axis=-1).astype(np.float32)  # (T, H, W, C)
        T, H, W, C = data.shape

        # Raw coordinates
        lats = ds.latitude.values.astype(np.float32)
        lons = ds.longitude.values.astype(np.float32)
        times = np.arange(T, dtype=np.float32)

        # Normalize coordinates to [0, 1]
        lat_norm  = (lats  - lats.min())  / (lats.max()  - lats.min()  + 1e-8)
        lon_norm  = (lons  - lons.min())  / (lons.max()  - lons.min()  + 1e-8)
        time_norm = times / (T - 1 + 1e-8)

        # Build meshgrid and flatten to (N, 3)
        lat_grid, lon_grid, t_grid = np.meshgrid(
            lat_norm, lon_norm, time_norm, indexing="ij"
        )
        self.coords = torch.from_numpy(
            np.stack([lat_grid, lon_grid, t_grid], axis=-1)
            .reshape(-1, 3)
        )

        # Flatten targets to (N, C)
        self.targets = torch.from_numpy(data.reshape(-1, C))

        # Z-score normalize targets
        if stats is None:
            self.mean = self.targets.mean(0)
            self.std  = self.targets.std(0) + 1e-8
        else:
            self.mean = stats["mean"]
            self.std  = stats["std"]

        self.targets = (self.targets - self.mean) / self.std

        print(f"  Dataset ready: {len(self)} samples | coords {self.coords.shape} | targets {self.targets.shape}")

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.coords[idx], self.targets[idx]

    @property
    def stats(self) -> Dict:
        return {"mean": self.mean, "std": self.std}

    def denormalize(self, targets: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to physical units."""
        return targets * self.std + self.mean


def make_dataloader(
    ds: xr.Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    stats: Optional[Dict] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Dict]:
    """
    Build a DataLoader from an xarray Dataset.

    Args:
        ds:          xarray Dataset (ERA5 or synthetic)
        batch_size:  samples per batch
        shuffle:     shuffle each epoch (True for train, False for val/test)
        stats:       normalization stats from training set (None = compute from ds)
        num_workers: parallel data loading workers (0 = main process, safe on Windows)

    Returns:
        (DataLoader, stats dict)
    """
    dataset = ClimateDataset(ds, stats=stats)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, dataset.stats


def split_dataset_by_time(
    ds: xr.Dataset,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """
    Split xarray Dataset along the time dimension.
    test_frac = 1 - train_frac - val_frac automatically.

    Args:
        ds:         full xarray Dataset
        train_frac: fraction of timesteps for training
        val_frac:   fraction of timesteps for validation

    Returns:
        (train_ds, val_ds, test_ds)
    """
    T = len(ds.time)
    t_train = int(T * train_frac)
    t_val   = int(T * (train_frac + val_frac))

    train_ds = ds.isel(time=slice(0, t_train))
    val_ds   = ds.isel(time=slice(t_train, t_val))
    test_ds  = ds.isel(time=slice(t_val, None))

    print(f"  Split: train={len(train_ds.time)} | val={len(val_ds.time)} | test={len(test_ds.time)} months")
    return train_ds, val_ds, test_ds