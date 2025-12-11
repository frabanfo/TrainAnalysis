import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

import dramatiq
from database.db_manager import DatabaseManager
from data_ingestion.trainstats_client import TrainStatsClient
from .dramatiq_config import TRAIN_QUEUE

@dramatiq.actor(queue_name=TRAIN_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def collect_train_data(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    try:
        logger.info(f"Starting train data collection: {start_date} to {end_date}")
        
        trainstats_client = TrainStatsClient()
        db_manager = DatabaseManager()
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        stations_df = db_manager.execute_query("SELECT station_code, station_name FROM stations ORDER BY station_name")
        
        if stations_df.empty:
            logger.error("No stations found in database")
            return {
                'task_type': 'train_collection',
                'chunk_id': chunk_id,
                'total_records': 0,
                'failed_dates': [],
                'success': False,
                'error': 'No stations found in database'
            }
        
        trainstats_stations = stations_df[['station_name', 'station_code']].values.tolist()
                   
        total_records = 0
        failed_dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                # Check if data for this date already exists
                existing_check = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM trains WHERE DATE(timestamp) = :check_date",
                    {'check_date': current_date.date()}
                )
                
                if not existing_check.empty and existing_check.iloc[0]['count'] > 0:
                    logger.info(f"Train data for {current_date.date()} already exists, skipping")
                    current_date += timedelta(days=1)
                    continue
                
                daily_data = []
                
                for station_name, station_code in trainstats_stations:
                    try:
                        logger.debug(f"Collecting data for station {station_name} for {current_date.date()}")
                        station_data = trainstats_client.get_station_data(station_name, station_code, current_date)
                        
                        if station_data:
                            daily_data.extend(station_data)
                            # Save raw data per station
                            trainstats_client.save_raw_data(station_data, current_date, station_name)
                            logger.info(f"Collected {len(station_data)} records from {station_name}")
                        else:
                            logger.debug(f"No data for {station_name} on {current_date.date()}")
                            
                    except Exception as e:
                        logger.error(f"Error collecting TrainStats data for {station_name}: {str(e)}")
                        continue
                
                # Store collected data
                if daily_data:
                    df = pd.DataFrame(daily_data)
                    
                    success = db_manager.store_train_data(df)
                    if success:
                        total_records += len(daily_data)
                        logger.info(f"Stored {len(daily_data)} train records for {current_date.date()} from TrainStats")
                    else:
                        failed_dates.append(current_date.date().isoformat())
                        logger.error(f"Failed to store data for {current_date.date()}")
                else:
                    logger.warning(f"No train data collected for {current_date.date()}")
                
                current_date += timedelta(days=1)
                
                # Memory cleanup
                del daily_data
                if 'df' in locals():
                    del df
                
            except Exception as e:
                logger.error(f"Error collecting train data for {current_date.date()}: {str(e)}")
                failed_dates.append(current_date.date().isoformat())
                current_date += timedelta(days=1)
        
        result = {
            'task_type': 'train_collection',
            'chunk_id': chunk_id,
            'total_records': total_records,
            'failed_dates': failed_dates,
            'stations_processed': len(trainstats_stations),
            'success': len(failed_dates) < (end_dt - start_dt).days * 0.5
        }
        
        logger.info(f"Train data collection completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Train data collection failed: {str(e)}")
        raise

