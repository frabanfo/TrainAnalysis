import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

import dramatiq
from src.database.db_manager import DatabaseManager
from src.data_ingestion.trainstats_client import TrainStatsClient
from src.data_quality.unified_processor import create_train_processor
from src.data_quality.metrics_store import DatabaseQualityMetricsStore
from .dramatiq_config import TRAIN_QUEUE
from .logging_config import setup_task_logging

@dramatiq.actor(queue_name=TRAIN_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def collect_train_data(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    log_file = setup_task_logging("train_collection", chunk_id)
    
    try:
        logger.info(f"Starting train data collection: {start_date} to {end_date} (chunk: {chunk_id})")
        
        trainstats_client = TrainStatsClient()
        db_manager = DatabaseManager()
        dq_processor = create_train_processor()
        metrics_store = DatabaseQualityMetricsStore(db_manager)
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        stations_df = db_manager.execute_query("SELECT station_code, station_name FROM stations ORDER BY station_name")
        
        if stations_df.empty:
            logger.error("No stations found in database")
            return {
                'task_type': 'train_collection_with_dq',
                'chunk_id': chunk_id,
                'total_records': 0,
                'failed_dates': [],
                'success': False,
                'error': 'No stations found in database'
            }
        
        trainstats_stations = stations_df[['station_name', 'station_code']].values.tolist()
        
        # DQ metrics tracking
        total_raw_records = 0
        total_clean_records = 0
        total_dropped_records = 0
        total_flagged_records = 0
        failed_dates = []
        dq_results = []
        skipped_dates = []
        
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
                    skipped_dates.append(current_date.date().isoformat())
                    current_date += timedelta(days=1)
                    continue
                
                daily_data = []
                
                # Collect raw data
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
                
                # Process collected data with DQ validation
                if daily_data:
                    raw_df = pd.DataFrame(daily_data)
                    total_raw_records += len(raw_df)
                    
                    logger.info(f"Starting DQ validation for {len(raw_df)} raw records on {current_date.date()}")
                    
                    # Apply data quality processing
                    clean_df, validation_results = dq_processor.process(raw_df)
                    
                    # Track DQ metrics
                    for result in validation_results:
                        total_dropped_records += result.dropped_records
                        total_flagged_records += result.flagged_records
                        dq_results.append(result)
                        
                        # Store validation results in metrics store
                        try:
                            metrics_store.store_validation_result(result)
                        except Exception as e:
                            logger.warning(f"Failed to store DQ metrics: {e}")
                    
                    total_clean_records += len(clean_df)
                    
                    # Log DQ summary for this date
                    logger.info(f"DQ Summary for {current_date.date()}: "
                              f"Raw: {len(raw_df)}, Clean: {len(clean_df)}, "
                              f"Dropped: {len(raw_df) - len(clean_df)}, "
                              f"Quality Rate: {len(clean_df)/len(raw_df)*100:.1f}%")
                    
                    # Store only clean data
                    if len(clean_df) > 0:
                        success = db_manager.store_train_data(clean_df)
                        if success:
                            logger.info(f"Stored {len(clean_df)} validated train records for {current_date.date()}")
                        else:
                            failed_dates.append(current_date.date().isoformat())
                            logger.error(f"Failed to store validated data for {current_date.date()}")
                    else:
                        logger.warning(f"No valid train data after DQ processing for {current_date.date()}")
                        failed_dates.append(current_date.date().isoformat())
                else:
                    logger.warning(f"No train data collected for {current_date.date()}")
                
                current_date += timedelta(days=1)
                
                # Memory cleanup
                del daily_data
                if 'raw_df' in locals():
                    del raw_df
                if 'clean_df' in locals():
                    del clean_df
                
            except Exception as e:
                logger.error(f"Error processing train data for {current_date.date()}: {str(e)}")
                failed_dates.append(current_date.date().isoformat())
                current_date += timedelta(days=1)
        
        # Calculate overall DQ metrics
        overall_quality_rate = total_clean_records / total_raw_records if total_raw_records > 0 else 0.0
        drop_rate = total_dropped_records / total_raw_records if total_raw_records > 0 else 0.0
        
        result = {
            'task_type': 'train_collection_with_dq',
            'chunk_id': chunk_id,
            'total_raw_records': int(total_raw_records),
            'total_clean_records': int(total_clean_records),
            'total_dropped_records': int(total_dropped_records),
            'total_flagged_records': int(total_flagged_records),
            'overall_quality_rate': float(overall_quality_rate),
            'drop_rate': float(drop_rate),
            'failed_dates': failed_dates,
            'stations_processed': int(len(trainstats_stations)),
            'dq_validation_count': int(len(dq_results)),
            'success': (total_raw_records > 0 and (total_clean_records > 0 or overall_quality_rate > 0.1)) or (len(skipped_dates) > 0 and len(failed_dates) == 0),
            'skipped_dates': skipped_dates
        }
        
        logger.info(f"Train data collection with DQ completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Train data collection with DQ failed: {str(e)}")
        return {
            'task_type': 'train_collection_with_dq',
            'chunk_id': chunk_id,
            'total_raw_records': 0,
            'total_clean_records': 0,
            'total_dropped_records': 0,
            'total_flagged_records': 0,
            'overall_quality_rate': 0.0,
            'drop_rate': 0.0,
            'failed_dates': [start_date, end_date],
            'skipped_dates': [],
            'stations_processed': 0,
            'dq_validation_count': 0,
            'success': False,
            'error': str(e)
        }