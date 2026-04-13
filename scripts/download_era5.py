"""
Download ERA5 data for all regions.

Usage:
    python scripts/download_era5.py --years 2000-2020
    python scripts/download_era5.py --regions north_america europe --years 2010-2020
    python scripts/download_era5.py --check-credentials
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.era5_loader import download_era5, _check_cds_credentials
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 climate data")
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["north_america", "europe", "asia_pacific"],
        choices=["north_america", "europe", "asia_pacific"],
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2000-2020",
        help="Year range e.g. 1980-2020",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save NetCDF files",
    )
    parser.add_argument(
        "--check-credentials",
        action="store_true",
        help="Check CDS API credentials and exit",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if file exists",
    )
    args = parser.parse_args()

    if args.check_credentials:
        if _check_cds_credentials():
            print("CDS API credentials found.")
            print("You can now download ERA5 data.")
        else:
            print("CDS API credentials NOT found.")
            print("Steps to set up:")
            print("  1. Register at https://cds.climate.copernicus.eu")
            print("  2. Get your API key from your profile page")
            print("  3. Create ~/.cdsapirc with:")
            print("     url: https://cds.climate.copernicus.eu/api/v2")
            print("     key: YOUR-UID:YOUR-API-KEY")
        return

    start, end = map(int, args.years.split("-"))
    years = list(range(start, end + 1))

    logger.info(f"Downloading ERA5 | regions={args.regions} | years={start}-{end}")

    for region in args.regions:
        path = download_era5(
            region=region,
            years=years,
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
        )
        if path:
            print(f"Downloaded: {path}")
        else:
            print(f"Failed: {region} — check credentials or network")


if __name__ == "__main__":
    main()