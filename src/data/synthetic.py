"""
Synthetic climate data generator for local development and testing.
Mimics ERA5 structure without needing a CDS API key.
Uses physically motivated signals: seasonal cycles, spatial gradients,
latitude-dependent temperature, and stochastic weather noise.
"""
import numpy as np
import xarray as xr
from typing import Optional


REGION_CONFIGS = {
    "north_america": {
        "lat_range": (15.0, 72.0),
        "lon_range": (-170.0, -50.0),
        "base_temp_k": 280.0,
        "precip_scale": 2.5e-4,
    },
    "europe": {
        "lat_range": (35.0, 72.0),
        "lon_range": (-25.0, 45.0),
        "base_temp_k": 278.0,
        "precip_scale": 2.0e-4,
    },
    "asia_pacific": {
        "lat_range": (-10.0, 60.0),
        "lon_range": (60.0, 180.0),
        "base_temp_k": 290.0,
        "precip_scale": 3.5e-4,
    },
}


def make_synthetic_era5(
    region: str,
    n_years: int = 10,
    spatial_resolution: float = 2.0,
    start_year: int = 1980,
    seed: int = 42,
) -> xr.Dataset:
    """
    Generate a synthetic ERA5-like xarray Dataset for a given region.

    Variables produced:
      t2m  - 2m temperature (K)
      tp   - total precipitation (m)
      sp   - surface pressure (Pa)
      u10  - 10m u-wind component (m/s)
      v10  - 10m v-wind component (m/s)

    Args:
        region: one of north_america, europe, asia_pacific
        n_years: number of years to simulate
        spatial_resolution: grid spacing in degrees
        start_year: first year of simulation
        seed: random seed for reproducibility

    Returns:
        xr.Dataset mimicking ERA5 monthly means structure
    """
    rng = np.random.default_rng(seed)
    cfg = REGION_CONFIGS[region]

    # Build coordinate axes
    lats = np.arange(cfg["lat_range"][0], cfg["lat_range"][1], spatial_resolution)
    lons = np.arange(cfg["lon_range"][0], cfg["lon_range"][1], spatial_resolution)
    n_months = n_years * 12
    times = np.arange(n_months)

    H, W, T = len(lats), len(lons), n_months

    # --- 2m Temperature (K) ---
    # Base temperature decreases with latitude (lapse rate proxy)
    lat_gradient = (cfg["lat_range"][1] - lats) / (cfg["lat_range"][1] - cfg["lat_range"][0])
    lat_effect = 20.0 * lat_gradient[None, :, None]  # warmer at low latitudes

    # Seasonal cycle: peaks in July (month 6), troughs in January (month 0)
    month_idx = np.arange(T) % 12
    seasonal = 12.0 * np.sin(2 * np.pi * (month_idx - 3) / 12)  # shape (T,)

    # Long-term warming trend (+0.02K/year)
    trend = 0.02 * (times / 12.0)

    # Stochastic noise
    noise = rng.normal(0, 1.5, size=(T, H, W))

    t2m = (
        cfg["base_temp_k"]
        + lat_effect.T                          # (T, H, W) broadcast
        + seasonal[:, None, None]
        + trend[:, None, None]
        + noise
    )

    # --- Total Precipitation (m) ---
    # Higher precip at mid-latitudes, seasonal modulation
    precip_lat = np.exp(-0.5 * ((lats - 45.0) / 20.0) ** 2)
    precip_seasonal = 1.0 + 0.4 * np.sin(2 * np.pi * (month_idx + 2) / 12)
    precip_noise = rng.exponential(cfg["precip_scale"], size=(T, H, W))

    tp = (
        cfg["precip_scale"]
        * precip_lat[None, :, None]
        * precip_seasonal[:, None, None]
        + precip_noise
    )
    tp = np.clip(tp, 0, None)

    # --- Surface Pressure (Pa) ---
    sp_base = 101325.0
    sp_noise = rng.normal(0, 800, size=(T, H, W))
    sp_seasonal = 300 * np.cos(2 * np.pi * month_idx / 12)
    sp = sp_base + sp_seasonal[:, None, None] + sp_noise

    # --- Wind Components (m/s) ---
    u10 = rng.normal(2.0, 4.0, size=(T, H, W))   # westerlies bias
    v10 = rng.normal(0.0, 3.0, size=(T, H, W))

    # Build time coordinate as datetime
    import pandas as pd
    time_index = pd.date_range(
        start=f"{start_year}-01-01",
        periods=n_months,
        freq="MS"
    )

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], t2m.astype(np.float32)),
            "tp":  (["time", "latitude", "longitude"], tp.astype(np.float32)),
            "sp":  (["time", "latitude", "longitude"], sp.astype(np.float32)),
            "u10": (["time", "latitude", "longitude"], u10.astype(np.float32)),
            "v10": (["time", "latitude", "longitude"], v10.astype(np.float32)),
        },
        coords={
            "time":      time_index,
            "latitude":  lats.astype(np.float32),
            "longitude": lons.astype(np.float32),
        },
        attrs={
            "description": f"Synthetic ERA5-like data for {region}",
            "region": region,
            "n_years": n_years,
            "start_year": start_year,
            "spatial_resolution_deg": spatial_resolution,
        }
    )

    return ds


def get_regional_datasets(
    regions: list,
    n_years: int = 10,
    spatial_resolution: float = 2.0,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic datasets for multiple regions.
    Each region gets a different seed so data is not identical.

    Returns:
        dict mapping region name -> xr.Dataset
    """
    return {
        region: make_synthetic_era5(
            region=region,
            n_years=n_years,
            spatial_resolution=spatial_resolution,
            seed=seed + i,
        )
        for i, region in enumerate(regions)
    }