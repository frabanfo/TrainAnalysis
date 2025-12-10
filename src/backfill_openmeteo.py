#!/usr/bin/env python3
"""
Backfill script for Open-Meteo weather data (Lombardy stations, 1 year)
"""

from datetime import datetime
from loguru import logger
import os
import sys

# Assicura che il root sia nel path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from data_ingestion.openmeteo_client import OpenMeteoClient


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/backfill_openmeteo_{time}.log", rotation="1 day", retention="30 days")
    logger.info("Starting Open-Meteo backfill...")


def main():
    setup_logging()

    # Path CSV stazioni ()
    #stations_csv = os.path.join(ROOT_DIR, "config", "stations_lombardia.csv")
    client = OpenMeteoClient()

    start = datetime(2025, 1, 1)
    end   = datetime(2025, 1, 30)

    logger.info(f"Backfill range: {start.date()} → {end.date()}")

    total_records = client.collect_data_streaming(
        start_date=start,
        end_date=end,
        chunk_days=7,    
        base_dir="data",
        save_raw=False   
    )

    logger.info(f"Backfill completed. Total records: {total_records}")


if __name__ == "__main__":
    main()
