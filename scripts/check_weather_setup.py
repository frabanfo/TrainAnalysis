#!/usr/bin/env python3
"""
Script per verificare che tutto sia configurato correttamente per la weather task.
"""

import os
import sys
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def check_database_connection():
    """Check database connection and tables"""
    try:
        from database.db_manager import DatabaseManager
        
        db = DatabaseManager()
        logger.info("✅ Database connection successful")
        
        # Check if weather table exists
        result = db.execute_query("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'weather'")
        if not result.empty and result.iloc[0]['count'] > 0:
            logger.info("✅ Weather table exists")
        else:
            logger.error("❌ Weather table not found")
            return False
            
        # Check stations
        stations = db.execute_query("SELECT COUNT(*) as count FROM stations")
        if not stations.empty:
            count = stations.iloc[0]['count']
            logger.info(f"✅ Found {count} stations in database")
        else:
            logger.error("❌ No stations found in database")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False

def check_openmeteo_client():
    """Check OpenMeteo client"""
    try:
        from data_ingestion.openmeteo_client import OpenMeteoClient
        
        client = OpenMeteoClient()
        logger.info("✅ OpenMeteo client created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenMeteo client check failed: {e}")
        return False

def check_dramatiq_config():
    """Check Dramatiq configuration"""
    try:
        from dramatiq_task.dramatiq_config import WEATHER_QUEUE, broker
        
        logger.info(f"✅ Dramatiq config loaded, weather queue: {WEATHER_QUEUE}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dramatiq config check failed: {e}")
        return False

def check_weather_tasks():
    """Check weather tasks import"""
    try:
        from dramatiq_task.weather_tasks import collect_weather_data, fetch_weather_chunk
        
        logger.info("✅ Weather tasks imported successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Weather tasks check failed: {e}")
        return False

def main():
    """Run all checks"""
    logger.info("🔍 Checking weather data collection setup...")
    
    checks = [
        ("Database Connection", check_database_connection),
        ("OpenMeteo Client", check_openmeteo_client),
        ("Dramatiq Config", check_dramatiq_config),
        ("Weather Tasks", check_weather_tasks),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        if not check_func():
            all_passed = False
    
    logger.info("\n" + "="*50)
    if all_passed:
        logger.info("🎉 All checks passed! Ready to run weather data collection.")
        logger.info("💡 Run: python scripts/test_weather_job.py")
    else:
        logger.error("❌ Some checks failed. Please fix the issues before running weather collection.")
    
    return all_passed

if __name__ == "__main__":
    main()