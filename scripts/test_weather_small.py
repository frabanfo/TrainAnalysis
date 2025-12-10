#!/usr/bin/env python3
"""
Test piccolo per weather data collection - solo 3 giorni per verificare che funzioni.
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dramatiq_task.weather_tasks import collect_weather_data


def main():
    """Test weather data collection for 3 days only"""
    
    # Setup logging
    logger.add("logs/test_weather_small_{time}.log", rotation="1 day")
    
    # Calculate date range - last 3 days only
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    logger.info(f"[TEST SMALL] Starting weather data collection test")
    logger.info(f"[TEST SMALL] Date range: {start_date.date()} → {end_date.date()}")
    logger.info(f"[TEST SMALL] This will collect weather data for all stations (3 days only)")
    
    # Send the weather collection task
    try:
        collect_weather_data.send(
            start_date.isoformat(),
            end_date.isoformat(),
            "test_weather_3_days"
        )
        
        logger.info("[TEST SMALL] Weather collection job enqueued successfully")
        logger.info("[TEST SMALL] Check the dramatiq worker logs to monitor progress")
        logger.info("[TEST SMALL] Data will be stored in the 'weather' table")
        
    except Exception as e:
        logger.error(f"[TEST SMALL] Failed to enqueue weather job: {e}")
        raise


if __name__ == "__main__":
    main()