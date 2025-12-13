import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

import dramatiq
try:
    from database.db_manager import DatabaseManager
    from data_ingestion.openmeteo_client import OpenMeteoClient
    from data_quality.unified_processor import create_weather_processor
    from data_quality.metrics_store import DatabaseQualityMetricsStore
    from .dramatiq_config import WEATHER_QUEUE
except ImportError:
    # Fallback for relative imports
    from ..database.db_manager import DatabaseManager
    from ..data_ingestion.openmeteo_client import OpenMeteoClient
    from ..data_quality.unified_processor import create_weather_processor
    from ..data_quality.metrics_store import DatabaseQualityMetricsStore
    from .dramatiq_config import WEATHER_QUEUE


@dramatiq.actor(queue_name=WEATHER_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def collect_weather_data(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    """
    Task Dramatiq per raccogliere dati meteo per tutte le stazioni in un range di date.
    """
    try:
        logger.info(f"Starting weather data collection: {start_date} to {end_date} (chunk_id: {chunk_id})")
        
        client = OpenMeteoClient()
        db_manager = DatabaseManager()
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Get all stations from DB
        stations_df = db_manager.execute_query(
            "SELECT station_code, station_name, latitude, longitude FROM stations"
        )
        
        if stations_df.empty:
            logger.error("No stations found in database")
            return {
                'task_type': 'weather_collection',
                'chunk_id': chunk_id,
                'total_records': 0,
                'failed_dates': [],
                'success': False,
                'error': 'No stations found'
            }
        
        logger.info(f"Found {len(stations_df)} stations")
        
        total_records = 0
        failed_dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                date_str = current_date.date().isoformat()
                logger.info(f"Processing date: {date_str}")
                
                # Check if weather data for this date already exists
                existing_check = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM weather WHERE DATE(timestamp) = :date",
                    {'date': current_date.date()}
                )
                
                if not existing_check.empty and existing_check.iloc[0]['count'] > 0:
                    logger.info(f"Weather data for {date_str} already exists, skipping")
                    current_date += timedelta(days=1)
                    continue
                
                # Collect weather data for all stations for this date
                daily_weather_records = []
                
                for _, station in stations_df.iterrows():
                    try:
                        station_dict = {
                            "code": station['station_code'],
                            "name": station['station_name'],
                            "lat": float(station['latitude']),
                            "lon": float(station['longitude'])
                        }
                        
                        # First check if CSV file already exists
                        year = current_date.year
                        csv_path = os.path.join(
                            "data", "curated", "openmeteo", station['station_code'], 
                            str(year), f"weather_{station['station_code']}_{current_date.date()}_{current_date.date()}.csv"
                        )
                        
                        # If file doesn't exist, fetch it
                        if not os.path.exists(csv_path):
                            records = client.fetch_station_chunk(
                                station=station_dict,
                                chunk_start=current_date.date(),
                                chunk_end=current_date.date(),
                                base_dir="data",
                                save_raw=False
                            )
                        
                        # Always try to read the CSV file (whether it was just created or already existed)
                        if os.path.exists(csv_path):
                            station_df = pd.read_csv(csv_path)
                            daily_weather_records.append(station_df)
                        else:
                            logger.warning(f"CSV file not found after fetch: {csv_path}")
                                
                    except Exception as e:
                        logger.error(f"Error processing weather for station {station['station_code']} on {date_str}: {e}")
                        continue
                
                # Store all weather data for this date
                if daily_weather_records:
                    combined_df = pd.concat(daily_weather_records, ignore_index=True)
                    
                    success = db_manager.store_weather_data(combined_df)
                    if success:
                        total_records += len(combined_df)
                        logger.info(f"Successfully stored {len(combined_df)} weather records for {date_str}")
                    else:
                        logger.error(f"Failed to store weather data for {date_str}")
                        failed_dates.append(date_str)
                else:
                    logger.warning(f"No weather data collected for {date_str}")
                    failed_dates.append(date_str)
                
                current_date += timedelta(days=1)
                
                # Memory cleanup
                if daily_weather_records:
                    del daily_weather_records, combined_df
                
            except Exception as e:
                logger.error(f"Error collecting weather data for {current_date.date()}: {str(e)}")
                failed_dates.append(current_date.date().isoformat())
                current_date += timedelta(days=1)
        
        result = {
            'task_type': 'weather_collection',
            'chunk_id': chunk_id,
            'total_records': total_records,
            'failed_dates': failed_dates,
            'success': len(failed_dates) < (end_dt - start_dt).days * 0.5
        }
        
        logger.info(f"Weather data collection completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Weather data collection failed: {str(e)}")
        raise


@dramatiq.actor(queue_name=WEATHER_QUEUE, max_retries=3)
def fetch_weather_chunk(
    station_code: str,
    station_name: str,
    lat: float,
    lon: float,
    chunk_start_iso: str,
    chunk_end_iso: str,
    base_dir: str = "data",
    save_raw: bool = False,
):
    """
    Task Dramatiq legacy per singola stazione (mantenuto per compatibilità).
    """
    logger.info(
        f"[WEATHER] {station_code} {station_name} — "
        f"{chunk_start_iso} → {chunk_end_iso}"
    )

    # 1. Ricostruisco stazione e date
    station = {
        "code": station_code,
        "name": station_name,
        "lat": float(lat),
        "lon": float(lon),
    }

    chunk_start = datetime.fromisoformat(chunk_start_iso).date()
    chunk_end = datetime.fromisoformat(chunk_end_iso).date()

    # 2. Creo il client e faccio fetch del chunk
    client = OpenMeteoClient()

    try:
        n_records = client.fetch_station_chunk(
            station=station,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            base_dir=base_dir,
            save_raw=save_raw,
        )
        logger.info(
            f"[WEATHER DONE] {station_code} {chunk_start_iso}→{chunk_end_iso} "
            f"-> {n_records} record"
        )
    except Exception as e:
        logger.error(
            f"[WEATHER ERROR] {station_code} {chunk_start_iso}→{chunk_end_iso}: {e}"
        )
        raise

@dramatiq.actor(queue_name=WEATHER_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def collect_weather_data_with_dq(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    """
    Enhanced weather data collection with integrated data quality validation.
    
    This task combines weather data collection with pre-integration data quality processing,
    ensuring only validated data is stored in the database.
    """
    try:
        logger.info(f"Starting weather data collection with DQ validation: {start_date} to {end_date} (chunk_id: {chunk_id})")
        
        client = OpenMeteoClient()
        db_manager = DatabaseManager()
        dq_processor = create_weather_processor()
        metrics_store = DatabaseQualityMetricsStore(db_manager)
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Get all stations from DB
        stations_df = db_manager.execute_query(
            "SELECT station_code, station_name, latitude, longitude FROM stations"
        )
        
        if stations_df.empty:
            logger.error("No stations found in database")
            return {
                'task_type': 'weather_collection_with_dq',
                'chunk_id': chunk_id,
                'total_records': 0,
                'failed_dates': [],
                'success': False,
                'error': 'No stations found'
            }
        
        logger.info(f"Found {len(stations_df)} stations")
        
        # DQ metrics tracking
        total_raw_records = 0
        total_clean_records = 0
        total_dropped_records = 0
        total_flagged_records = 0
        failed_dates = []
        dq_results = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                date_str = current_date.date().isoformat()
                logger.info(f"Processing date: {date_str}")
                
                # Check if weather data for this date already exists
                existing_check = db_manager.execute_query(
                    "SELECT COUNT(*) as count FROM weather WHERE DATE(timestamp) = :date",
                    {'date': current_date.date()}
                )
                
                if not existing_check.empty and existing_check.iloc[0]['count'] > 0:
                    logger.info(f"Weather data for {date_str} already exists, skipping")
                    current_date += timedelta(days=1)
                    continue
                
                # Collect weather data for all stations for this date
                daily_weather_records = []
                
                for _, station in stations_df.iterrows():
                    try:
                        station_dict = {
                            "code": station['station_code'],
                            "name": station['station_name'],
                            "lat": float(station['latitude']),
                            "lon": float(station['longitude'])
                        }
                        
                        # First check if CSV file already exists
                        year = current_date.year
                        csv_path = os.path.join(
                            "data", "curated", "openmeteo", station['station_code'], 
                            str(year), f"weather_{station['station_code']}_{current_date.date()}_{current_date.date()}.csv"
                        )
                        
                        # If file doesn't exist, fetch it
                        if not os.path.exists(csv_path):
                            records = client.fetch_station_chunk(
                                station=station_dict,
                                chunk_start=current_date.date(),
                                chunk_end=current_date.date(),
                                base_dir="data",
                                save_raw=False
                            )
                        
                        # Always try to read the CSV file (whether it was just created or already existed)
                        if os.path.exists(csv_path):
                            station_df = pd.read_csv(csv_path)
                            daily_weather_records.append(station_df)
                        else:
                            logger.warning(f"CSV file not found after fetch: {csv_path}")
                                
                    except Exception as e:
                        logger.error(f"Error processing weather for station {station['station_code']} on {date_str}: {e}")
                        continue
                
                # Process collected data with DQ validation
                if daily_weather_records:
                    raw_df = pd.concat(daily_weather_records, ignore_index=True)
                    total_raw_records += len(raw_df)
                    
                    logger.info(f"Starting DQ validation for {len(raw_df)} raw weather records on {date_str}")
                    
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
                    logger.info(f"Weather DQ Summary for {date_str}: "
                              f"Raw: {len(raw_df)}, Clean: {len(clean_df)}, "
                              f"Dropped: {len(raw_df) - len(clean_df)}, "
                              f"Quality Rate: {len(clean_df)/len(raw_df)*100:.1f}%")
                    
                    # Store only clean data
                    if len(clean_df) > 0:
                        success = db_manager.store_weather_data(clean_df)
                        if success:
                            logger.info(f"Successfully stored {len(clean_df)} validated weather records for {date_str}")
                        else:
                            logger.error(f"Failed to store weather data for {date_str}")
                            failed_dates.append(date_str)
                    else:
                        logger.warning(f"No valid weather data after DQ processing for {date_str}")
                        failed_dates.append(date_str)
                else:
                    logger.warning(f"No weather data collected for {date_str}")
                    failed_dates.append(date_str)
                
                current_date += timedelta(days=1)
                
                # Memory cleanup
                if daily_weather_records:
                    del daily_weather_records
                if 'raw_df' in locals():
                    del raw_df
                if 'clean_df' in locals():
                    del clean_df
                
            except Exception as e:
                logger.error(f"Error collecting weather data for {current_date.date()}: {str(e)}")
                failed_dates.append(current_date.date().isoformat())
                current_date += timedelta(days=1)
        
        # Calculate overall DQ metrics
        overall_quality_rate = total_clean_records / total_raw_records if total_raw_records > 0 else 0.0
        drop_rate = total_dropped_records / total_raw_records if total_raw_records > 0 else 0.0
        
        result = {
            'task_type': 'weather_collection_with_dq',
            'chunk_id': chunk_id,
            'total_raw_records': total_raw_records,
            'total_clean_records': total_clean_records,
            'total_dropped_records': total_dropped_records,
            'total_flagged_records': total_flagged_records,
            'overall_quality_rate': overall_quality_rate,
            'drop_rate': drop_rate,
            'failed_dates': failed_dates,
            'dq_validation_count': len(dq_results),
            'success': len(failed_dates) < (end_dt - start_dt).days * 0.5 and overall_quality_rate > 0.5
        }
        
        logger.info(f"Weather data collection with DQ completed: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Weather data collection with DQ failed: {str(e)}")
        raise