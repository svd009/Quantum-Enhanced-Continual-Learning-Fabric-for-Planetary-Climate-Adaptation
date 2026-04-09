import pytest
import torch
from src.data.synthetic import make_synthetic_era5, get_regional_datasets
from src.data.preprocessing import make_dataloader, split_dataset_by_time


def test_synthetic_dataset_shape():
    ds = make_synthetic_era5("europe", n_years=2)
    assert "t2m" in ds
    assert "tp" in ds
    assert len(ds.time) == 24


def test_all_regions_generate():
    datasets = get_regional_datasets(
        ["north_america", "europe", "asia_pacific"], n_years=2
    )
    assert len(datasets) == 3
    for name, ds in datasets.items():
        assert len(ds.time) == 24


def test_split_sizes():
    ds = make_synthetic_era5("north_america", n_years=10)
    train, val, test = split_dataset_by_time(ds, train_frac=0.7, val_frac=0.15)
    total = len(train.time) + len(val.time) + len(test.time)
    assert total == len(ds.time)


def test_dataloader_batch_shape():
    ds = make_synthetic_era5("europe", n_years=2)
    train_ds, _, _ = split_dataset_by_time(ds)
    loader, stats = make_dataloader(train_ds, batch_size=16)
    coords, targets = next(iter(loader))
    assert coords.shape[1] == 3
    assert targets.shape[1] == 5
    assert "mean" in stats
    assert "std" in stats


def test_normalization_stats_shared():
    ds = make_synthetic_era5("europe", n_years=3)
    train_ds, val_ds, _ = split_dataset_by_time(ds)
    train_loader, stats = make_dataloader(train_ds, batch_size=16)
    val_loader, _ = make_dataloader(val_ds, batch_size=16, stats=stats)
    _, targets = next(iter(val_loader))
    assert targets.shape[1] == 5