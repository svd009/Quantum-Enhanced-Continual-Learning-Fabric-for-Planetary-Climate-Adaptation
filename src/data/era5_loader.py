"""
ERA5 climate data downloader and loader.
Uses the Copernicus Climate Data Store (CDS) API.

Setup:
    1. Register free account at https://cds.climate.copernicus.eu
    2. Get your API key from your profile page
    3. Create file ~/.cdsapirc with:
       url: https://cds.climate.copernicus.eu/api/v2
       key: YOUR-UID:YOUR-API-KEY

    Or set environment variables:
       CDSAPI_URL=https://cds.climate.copernicus.eu/api/v2
       CDSAPI_KEY=YOUR-UID:YOUR-API-KEY
"""
import os
import xarray as xr
from pathlib import Path
from typing import List, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)

VARIABLES = [
    "2m_temperature",
    "total_precipitation",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

REGIONS = {
    "north_america": {"area": [72, -170, 15, -50]},
    "europe":        {"area": [72,  -25, 35,  45]},
    "asia_pacific":  {"area": [60,   60, -10, 180]},
}


def _check_cds_credentials() -> bool:
    """Check if CDS API credentials are available."""
    rc_file = Path.home() / ".cdsapirc"
    has_env = bool(os.environ.get("CDSAPI_KEY"))
    has_file = rc_file.exists()

    if not has_env and not has_file:
        logger.warning(
            "No CDS API credentials found. "
            "Set CDSAPI_KEY env var or create ~/.cdsapirc. "
            "See https://cds.climate.copernicus.eu/api-how-to"
        )
        return False
    return True


def download_era5(
    region: str,
    years: List[int],
    output_dir: Path,
    variables: List[str] = VARIABLES,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Download ERA5 monthly means for a region and year range.

    Args:
        region:     one of north_america, europe, asia_pacific
        years:      list of years to download
        output_dir: directory to save NetCDF files
        variables:  ERA5 variable names to download
        overwrite:  re-download even if file exists

    Returns:
        Path to downloaded NetCDF file, or None if download failed
    """
    if not _check_cds_credentials():
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / f"era5_{region}_{years[0]}_{years[-1]}.nc"

    if out_path.exists() and not overwrite:
        logger.info(f"ERA5 file already exists: {out_path}")
        return out_path

    try:
        import cdsapi
    except ImportError:
        logger.error("cdsapi not installed. Run: pip install cdsapi")
        return None

    try:
        client = cdsapi.Client()
        logger.info(f"Downloading ERA5 | region={region} | years={years[0]}-{years[-1]}")

        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable":     variables,
                "year":         [str(y) for y in years],
                "month":        [f"{m:02d}" for m in range(1, 13)],
                "time":         "00:00",
                "area":         REGIONS[region]["area"],
                "format":       "netcdf",
            },
            str(out_path),
        )
        logger.info(f"Downloaded: {out_path}")
        return out_path

    except Exception as e:
        logger.error(f"ERA5 download failed: {e}")
        return None


def load_era5(path: Path) -> Optional[xr.Dataset]:
    """
    Load ERA5 NetCDF file into xarray Dataset.

    Args:
        path: path to NetCDF file

    Returns:
        xarray Dataset or None if file not found
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    try:
        ds = xr.open_dataset(path, engine="netcdf4")
        logger.info(f"Loaded ERA5: {dict(ds.dims)} | vars={list(ds.data_vars)}")
        return ds
    except Exception as e:
        logger.error(f"Failed to load ERA5 file: {e}")
        return None


def load_or_generate(
    region: str,
    years: List[int],
    data_dir: Path,
    fallback_synthetic: bool = True,
    n_years_synthetic: int = 10,
) -> xr.Dataset:
    """
    Try to load real ERA5 data, fall back to synthetic if unavailable.
    This is the main entry point used by the training pipeline.

    Args:
        region:              region name
        years:               list of years
        data_dir:            directory with ERA5 files
        fallback_synthetic:  use synthetic data if ERA5 unavailable
        n_years_synthetic:   years of synthetic data to generate

    Returns:
        xarray Dataset (real or synthetic)
    """
    data_dir = Path(data_dir)
    nc_path  = data_dir / f"era5_{region}_{years[0]}_{years[-1]}.nc"

    if nc_path.exists():
        logger.info(f"Loading real ERA5 data: {nc_path}")
        ds = load_era5(nc_path)
        if ds is not None:
            return ds

    if _check_cds_credentials():
        logger.info(f"Attempting ERA5 download for {region}...")
        path = download_era5(region, years, data_dir)
        if path is not None:
            ds = load_era5(path)
            if ds is not None:
                return ds

    if fallback_synthetic:
        logger.warning(
            f"ERA5 unavailable for {region}. "
            f"Using synthetic data ({n_years_synthetic} years). "
            f"To use real data: set up CDS API credentials."
        )
        from .synthetic import make_synthetic_era5
        return make_synthetic_era5(region=region, n_years=n_years_synthetic)

    raise RuntimeError(
        f"ERA5 data unavailable for {region} and synthetic fallback is disabled."
    )


def get_era5_or_synthetic(
    regions: List[str],
    years: List[int],
    data_dir: str = "data/raw",
    fallback_synthetic: bool = True,
    n_years_synthetic: int = 10,
) -> dict:
    """
    Load ERA5 (or synthetic fallback) for multiple regions.
    Drop-in replacement for get_regional_datasets() in the training pipeline.

    Returns:
        dict mapping region name -> xr.Dataset
    """
    return {
        region: load_or_generate(
            region=region,
            years=years,
            data_dir=Path(data_dir),
            fallback_synthetic=fallback_synthetic,
            n_years_synthetic=n_years_synthetic,
        )
        for region in regions
    }