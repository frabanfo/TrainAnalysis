import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

import dramatiq
from database.db_manager import DatabaseManager
from data_ingestion.openmeteo_client import OpenMeteoClient
from .dramatiq_config import WEATHER_QUEUE


@dramatiq.actor(queue_name=WEATHER_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000)
def collect_weather_data(start_date: str, end_date: str, chunk_id: str = None) -> Dict[str, Any]:
    """
    Task Dramatiq per raccogliere dati meteo per tutte le stazioni in un range di date.
    Segue il pattern della train_task con controllo DB e storage.
    """
    try:
        logger.info(f"Starting weather data collection: {start_date} to {end_date}")
        
        client = OpenMeteoClient()
        db_manager = DatabaseManager()
        
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Get all stations from DB
        stations_df = db_manager.execute_query(
            "SELECT station_code, station_name, latitude, longitude FROM stations"
        )
        
        if stations_df.empty:
            logger.warning("No stations found in database")
            return {
                'task_type': 'weather_collection',
                'chunk_id': chunk_id,
                'total_records': 0,
                'failed_dates': [],
                'success': False,
                'error': 'No stations found'
            }
        
        total_records = 0
        failed_dates = []
        current_date = start_dt
        
        while current_date <= end_dt:
            try:
                date_str = current_date.date().isoformat()
                
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
                        
                        # Fetch weather data for this station and date
                        records = client.fetch_station_chunk(
                            station=station_dict,
                            chunk_start=current_date.date(),
                            chunk_end=current_date.date(),
                            base_dir="data",
                            save_raw=False
                        )
                        
                        if records > 0:
                            # Read the generated CSV to get the actual data
                            year = current_date.year
                            csv_path = os.path.join(
                                "data", "curated", "openmeteo", station['station_code'], 
                                str(year), f"weather_{station['station_code']}_{current_date.date()}_{current_date.date()}.csv"
                            )
                            
                            if os.path.exists(csv_path):
                                station_df = pd.read_csv(csv_path)
                                daily_weather_records.append(station_df)
                                
                    except Exception as e:
                        logger.error(f"Error fetching weather for station {station['station_code']} on {date_str}: {e}")
                        continue
                
                # Store all weather data for this date
                if daily_weather_records:
                    combined_df = pd.concat(daily_weather_records, ignore_index=True)
                    
                    success = db_manager.store_weather_data(combined_df)
                    if success:
                        total_records += len(combined_df)
                        logger.info(f"Stored {len(combined_df)} weather records for {date_str}")
                    else:
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
