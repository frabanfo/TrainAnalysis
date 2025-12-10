#!/usr/bin/env python3
"""
Job di test per la pipeline meteo:
raccoglie l'ultimo mese di dati meteo per tutte le stazioni.
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dramatiq_task.weather_tasks import collect_weather_data


def main():
    """Test weather data collection for the last month"""
    
    # Setup logging
    logger.add("logs/test_weather_{time}.log", rotation="1 day")
    
    # Calculate date range - last month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"[TEST] Starting weather data collection test")
    logger.info(f"[TEST] Date range: {start_date.date()} → {end_date.date()}")
    logger.info(f"[TEST] This will collect weather data for all stations in the database")
    
    # Send the weather collection task
    try:
        collect_weather_data.send(
            start_date.isoformat(),
            end_date.isoformat(),
            "test_weather_last_month"
        )
        
        logger.info("[TEST] Weather collection job enqueued successfully")
        logger.info("[TEST] Check the dramatiq worker logs to monitor progress")
        logger.info("[TEST] Data will be stored in the 'weather' table")
        
    except Exception as e:
        logger.error(f"[TEST] Failed to enqueue weather job: {e}")
        raise


def test_single_station():
    """Test weather data collection for a single station (legacy method)"""
    from dramatiq_task.weather_tasks import fetch_weather_chunk
    
    # Milano Centrale - test con una settimana
    station_code = "S02430"
    station_name = "Milano Centrale"
    lat = 45.4842
    lon = 9.2044

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    logger.info(f"[TEST SINGLE] Testing single station: {station_code}")
    logger.info(f"[TEST SINGLE] Date range: {start_date.date()} → {end_date.date()}")

    fetch_weather_chunk.send(
        station_code,
        station_name,
        lat,
        lon,
        start_date.date().isoformat(),
        end_date.date().isoformat(),
        "data",
        False,
    )

    logger.info("[TEST SINGLE] Single station job enqueued successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test weather data collection')
    parser.add_argument('--single', action='store_true', 
                       help='Test single station instead of all stations')
    
    args = parser.parse_args()
    
    if args.single:
        test_single_station()
    else:
        main()
