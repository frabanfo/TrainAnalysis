import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from ..database.db_manager import DatabaseManager
from .feature_engineer import FeatureEngineer


class TrainWeatherIntegrator:
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.feature_engineer = FeatureEngineer()
        
        self.config = {
            'time_tolerance_minutes': 60,  # Match weather within 30 minutes of train time
            'batch_size': 1000,  # Process in batches for memory efficiency
            'enable_feature_engineering': True
        }
    
    def integrate_data(self, start_date: str, end_date: str, 
                      integration_id: str = None) -> Dict[str, Any]:
        integration_id = integration_id or f"integration_{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting train-weather integration: {start_date} to {end_date}")
        
        start_time = datetime.now()
        
        try:
            existing_check = self._check_existing_integration(start_date, end_date)
            if existing_check['exists']:
                logger.info(f"Integration data for {start_date} to {end_date} already exists, skipping")
                return {
                    'integration_id': integration_id,
                    'success': True,
                    'start_date': start_date,
                    'end_date': end_date,
                    'source_data': {'train_records': 0, 'weather_records': 0},
                    'integrated_data': {
                        'total_records': existing_check['record_count'],
                        'match_rate': 0.0,
                        'quality_score': 1.0
                    },
                    'performance': {'duration_seconds': 0, 'records_per_second': 0},
                    'storage_success': True,
                    'skipped': True,
                    'reason': 'Data already exists'
                }
            
            train_data, weather_data = self._load_source_data(start_date, end_date)
            
            if train_data.empty or weather_data.empty:
                return self._empty_integration_result(integration_id, "No source data available")
            
            integrated_data = self._perform_integration(train_data, weather_data, integration_id)
            
            if integrated_data.empty:
                return self._empty_integration_result(integration_id, "Integration produced no results")
            
            # Step 3: Feature engineering
            if self.config['enable_feature_engineering']:
                integrated_data = self.feature_engineer.engineer_features(integrated_data)
            
            storage_success = self._store_integrated_data(integrated_data, integration_id)
            
            integration_duration = (datetime.now() - start_time).total_seconds()
            
            # Calculate match rate from integration results
            match_rate = 0.0
            quality_score = 1.0
            
            if not integrated_data.empty:
                # Count records with weather data
                weather_matched = integrated_data['temperature'].notna().sum()
                match_rate = weather_matched / len(integrated_data) if len(integrated_data) > 0 else 0.0
                quality_score = min(match_rate + 0.2, 1.0)  # Base quality score on match rate
            
            return {
                'integration_id': integration_id,
                'success': True,
                'start_date': start_date,
                'end_date': end_date,
                'source_data': {
                    'train_records': len(train_data),
                    'weather_records': len(weather_data)
                },
                'integrated_data': {
                    'total_records': len(integrated_data),
                    'match_rate': match_rate,
                    'quality_score': quality_score
                },
                'performance': {
                    'duration_seconds': integration_duration,
                    'records_per_second': len(integrated_data) / integration_duration if integration_duration > 0 else 0
                },
                'storage_success': storage_success
            }
            
        except Exception as e:
            integration_duration = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"Integration failed for {integration_id}: {str(e)}")
            
            return {
                'integration_id': integration_id,
                'success': False,
                'error': str(e),
                'source_data': {'train_records': 0, 'weather_records': 0},
                'integrated_data': {'total_records': 0, 'match_rate': 0.0, 'quality_score': 0.0},
                'performance': {'duration_seconds': integration_duration, 'records_per_second': 0},
                'storage_success': False
            }
    
    def _load_source_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading source data for integration")
        
        train_query = """
        SELECT 
            train_id, timestamp, station_code, scheduled_time, actual_time,
            delay_minutes, train_category, route, delay_status, destination, is_cancelled
        FROM trains 
        WHERE DATE(timestamp) >= DATE(:start_date) AND DATE(timestamp) <= DATE(:end_date)
        ORDER BY timestamp, station_code
        """
        
        weather_query = """
        SELECT 
            station_code, timestamp, temperature, wind_speed, precip_mm, weather_code
        FROM weather 
        WHERE DATE(timestamp) >= DATE(:start_date) AND DATE(timestamp) <= DATE(:end_date)
        ORDER BY timestamp, station_code
        """
        
        params = {'start_date': start_date, 'end_date': end_date}
        
        train_data = self.db_manager.execute_query(train_query, params)
        weather_data = self.db_manager.execute_query(weather_query, params)
        
        if not train_data.empty:
            train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
        if not weather_data.empty:
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
        
        logger.info(f"Loaded {len(train_data)} train records and {len(weather_data)} weather records")
        
        return train_data, weather_data
    
    def _check_existing_integration(self, start_date: str, end_date: str) -> Dict[str, Any]:
        try:
            existing_query = """
            SELECT COUNT(*) as count, MIN(DATE(timestamp)) as min_date, MAX(DATE(timestamp)) as max_date
            FROM train_weather_integrated 
            WHERE DATE(timestamp) >= DATE(:start_date) AND DATE(timestamp) <= DATE(:end_date)
            """
            
            params = {'start_date': start_date, 'end_date': end_date}
            result = self.db_manager.execute_query(existing_query, params)
            
            if result.empty:
                return {'exists': False, 'record_count': 0}
            
            record_count = result.iloc[0]['count']
            
            if record_count > 0:
                logger.info(f"Found {record_count} existing integration records for date range {start_date} to {end_date}")
                return {'exists': True, 'record_count': record_count}
            else:
                return {'exists': False, 'record_count': 0}
                
        except Exception as e:
            logger.error(f"Error checking existing integration data: {str(e)}")
            return {'exists': False, 'record_count': 0}
    
    def check_existing_integration_by_date(self, check_date: str) -> bool:
        try:
            existing_check = self.db_manager.execute_query(
                "SELECT COUNT(*) as count FROM train_weather_integrated WHERE DATE(timestamp) = :check_date",
                {'check_date': check_date}
            )
            
            if not existing_check.empty and existing_check.iloc[0]['count'] > 0:
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking existing integration data for {check_date}: {str(e)}")
            return False
    
    def _perform_integration(self, train_data: pd.DataFrame, weather_data: pd.DataFrame,
                           integration_id: str) -> pd.DataFrame:

        logger.info(f"Performing train-weather integration for {integration_id}")
        
        if train_data.empty or weather_data.empty:
            logger.warning("Cannot integrate: empty source data")
            return pd.DataFrame()
        
        time_tolerance = pd.Timedelta(minutes=self.config['time_tolerance_minutes'])
        
        integrated_records = []
        unmatched_trains = 0
        
        batch_size = self.config['batch_size']
        total_batches = (len(train_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_data))
            train_batch = train_data.iloc[start_idx:end_idx]
            
            logger.debug(f"Processing integration batch {batch_idx + 1}/{total_batches}")
            
            for _, train_row in train_batch.iterrows():
                weather_match = self._find_weather_match(train_row, weather_data, time_tolerance)
                
                if weather_match is not None:
                    integrated_record = self._create_integrated_record(train_row, weather_match)
                    integrated_records.append(integrated_record)
                else:
                    unmatched_trains += 1
                    integrated_record = self._create_integrated_record(train_row, None)
                    integrated_records.append(integrated_record)
        
        if not integrated_records:
            logger.warning("No integrated records created")
            return pd.DataFrame()
        
        integrated_df = pd.DataFrame(integrated_records)
        
        match_rate = (len(integrated_records) - unmatched_trains) / len(integrated_records) if integrated_records else 0
        
        logger.info(f"Integration completed: {len(integrated_records)} records, {match_rate:.2%} match rate")
        
        return integrated_df
    
    def _find_weather_match(self, train_row: pd.Series, weather_data: pd.DataFrame, 
                          time_tolerance: pd.Timedelta) -> Optional[pd.Series]:
        
        # Filter by station and time window
        station_weather = weather_data[
            (weather_data['station_code'] == train_row['station_code']) &
            (weather_data['timestamp'] >= train_row['timestamp'] - time_tolerance) &
            (weather_data['timestamp'] <= train_row['timestamp'] + time_tolerance)
        ]
        
        if station_weather.empty:
            return None
        
        # Find closest time match
        station_weather = station_weather.copy()
        station_weather['time_diff'] = abs(station_weather['timestamp'] - train_row['timestamp'])
        closest_match = station_weather.loc[station_weather['time_diff'].idxmin()]
        
        return closest_match
    
    def _create_integrated_record(self, train_row: pd.Series, 
                                weather_row: Optional[pd.Series]) -> Dict[str, Any]:
        
        record = {
            'train_id': train_row['train_id'],
            'timestamp': train_row['timestamp'],
            'station_code': train_row['station_code'],
            'delay_minutes': train_row.get('delay_minutes'),
            'train_category': train_row.get('train_category'),
            'route': train_row.get('route'),
            'delay_status': train_row.get('delay_status'),
            'destination': train_row.get('destination'),
            'is_cancelled': train_row.get('is_cancelled', False)
        }
        
        if weather_row is not None:
            record.update({
                'temperature': weather_row.get('temperature'),
                'wind_speed': weather_row.get('wind_speed'),
                'precip_mm': weather_row.get('precip_mm'),
                'weather_code': weather_row.get('weather_code')
            })
        else:
            record.update({
                'temperature': None,
                'wind_speed': None,
                'precip_mm': None,
                'weather_code': None
            })
        
        return record
    
    def _store_integrated_data(self, integrated_data: pd.DataFrame, 
                             integration_id: str) -> bool:
        if integrated_data.empty:
            logger.warning("No integrated data to store")
            return False
        
        try:
            logger.info(f"Storing {len(integrated_data)} integrated records")
            
            # Get the expected columns from the database table
            expected_columns = [
                'train_id', 'timestamp', 'station_code', 'delay_minutes', 'temperature',
                'wind_speed', 'precip_mm', 'weather_code', 'train_category', 'route',
                'delay_status', 'destination', 'is_cancelled', 'hour_of_day', 'day_of_week',
                'is_weekend', 'is_rush_hour', 'temp_category', 'is_raining', 'rain_intensity',
                'wind_category', 'is_delayed', 'delay_category'
            ]
            
            # Filter the dataframe to only include columns that exist in the database
            available_columns = [col for col in expected_columns if col in integrated_data.columns]
            filtered_data = integrated_data[available_columns].copy()
            
            logger.info(f"Storing {len(available_columns)} columns: {available_columns}")
            
            # Use smaller batch size for database insertion to avoid SQL query limits
            db_batch_size = 100  # Smaller batch size for database operations
            total_stored = 0
            
            for i in range(0, len(filtered_data), db_batch_size):
                batch = filtered_data.iloc[i:i+db_batch_size]
                
                # Use 'multi' method but with smaller batches
                batch.to_sql(
                    'train_weather_integrated',
                    self.db_manager.engine,
                    if_exists='append',
                    index=False,
                    method='multi'
                )
                total_stored += len(batch)
                
                if i % (db_batch_size * 10) == 0:  # Log progress every 1000 records
                    logger.info(f"Stored {total_stored}/{len(filtered_data)} records")
            
            logger.info(f"Successfully stored {total_stored} integrated records")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store integrated data: {str(e)}")
            logger.error(f"Data shape: {integrated_data.shape}")
            logger.error(f"Columns: {list(integrated_data.columns)}")
            return False
    

    
    def _empty_integration_result(self, integration_id: str, reason: str) -> Dict[str, Any]:
        logger.warning(f"Empty integration result for {integration_id}: {reason}")
        
        return {
            'integration_id': integration_id,
            'success': False,
            'error': reason,
            'source_data': {'train_records': 0, 'weather_records': 0},
            'integrated_data': {'total_records': 0, 'match_rate': 0.0, 'quality_score': 0.0},
            'performance': {'duration_seconds': 0, 'records_per_second': 0},
            'storage_success': False
        }
    
