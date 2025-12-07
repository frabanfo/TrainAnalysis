#!/usr/bin/env python3
"""
Initialization script to fetch stations when database is ready
this will run automatically when you run 'docker compose up'
"""

import os
import sys
import time
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/app/src')

def wait_for_database():
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            from database.db_manager import DatabaseManager
            db_manager = DatabaseManager()
            
            test_query = "SELECT 1"
            result = db_manager.execute_query(test_query)
            
            if not result.empty:
                logger.info("Database is ready!")
                return True
                
        except Exception as e:
            logger.info(f"Database not ready yet (attempt {retry_count + 1}/{max_retries}): {str(e)}")
            retry_count += 1
            time.sleep(5)
    
    logger.error("Database failed to become ready within timeout")
    return False

def check_stations_exist():
    """Check if Lombardia stations are already in database"""
    try:
        from database.db_manager import DatabaseManager
        db_manager = DatabaseManager()
        
        # Check if we have a reasonable number of stations
        query = "SELECT COUNT(*) as count FROM stations"
        result = db_manager.execute_query(query)
        
        if not result.empty:
            count = result.iloc[0]['count']
            logger.info(f"Found {count} stations in database")
            
            # If we have more than 300 stations, assume Lombardia stations are loaded
            if count > 300:
                logger.info("Lombardia stations appear to already be loaded")
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Error checking existing stations: {str(e)}")
        return False

def fetch_lombardia_stations():
    """Fetch and store Lombardia stations"""
    try:
        from data_ingestion.fetch_lombardia_stations import LombardiaStationsFetcher
        
        logger.info("Starting Lombardia stations initialization...")
        
        fetcher = LombardiaStationsFetcher()
        success = fetcher.run()
        
        if success:
            logger.info("Lombardia stations initialization completed successfully!")
            return True
        else:
            logger.error("Lombardia stations initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during Lombardia stations fetch: {str(e)}")
        return False

def main():
    """Main initialization function"""
    logger.info("Starting Lombardia stations initialization...")
    
    if not wait_for_database():
        logger.error("Database initialization timeout")
        sys.exit(1)
    
    if check_stations_exist():
        logger.info("Lombardia stations already initialized, skipping fetch")
        return
    
    success = fetch_lombardia_stations()
    
    if success:
        logger.info("Lombardia stations initialization completed!")
    else:
        logger.error("Lombardia stations initialization failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()