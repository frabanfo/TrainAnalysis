import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

import dramatiq
from database.db_manager import DatabaseManager
from data_ingestion.viaggiotreno_client import ViaggiotrenoClient
from .dramatiq_config import TRAIN_QUEUE

@dramatiq.actor(queue_name=TRAIN_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000)
def collect_train_data(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    try:
        logger.info(f"Starting train data collection: {start_date} to {end_date}")
        
        client = ViaggiotrenoClient()
        db_manager = DatabaseManager()
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        total_records = 0
        failed_dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                # Check if data for this date already exists
                existing_check = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM trains WHERE created_at = :created_at",
                    {'created_at': current_date.date()}
                )
                
                if not existing_check.empty and existing_check.iloc[0]['count'] > 0:
                    logger.info(f"Train data for {current_date.date()} already exists, skipping")
                    current_date += timedelta(days=1)
                    continue
                
                # Collect daily data
                daily_data = client._collect_daily_data(current_date)
                
                if daily_data:
                    df = pd.DataFrame(daily_data)
                    
                    success = db_manager.store_train_data(df)
                    if success:
                        total_records += len(daily_data)
                        logger.info(f"Stored {len(daily_data)} train records for {current_date.date()}")
                    else:
                        failed_dates.append(current_date.date().isoformat())
                else:
                    logger.warning(f"No train data for {current_date.date()}")
                
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
            'success': len(failed_dates) < (end_dt - start_dt).days * 0.5
        }
        
        logger.info(f"Train data collection completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Train data collection failed: {str(e)}")
        raise

