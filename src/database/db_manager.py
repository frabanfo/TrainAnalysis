import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
import os
from typing import List, Dict, Any, Optional
from loguru import logger

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self):
        self.host = os.getenv('POSTGRES_HOST', 'localhost')
        self.database = os.getenv('POSTGRES_DB', 'railway_analysis')
        self.user = os.getenv('POSTGRES_USER', 'railway_user')
        self.password = os.getenv('POSTGRES_PASSWORD', 'railway_pass')
        self.port = os.getenv('POSTGRES_PORT', '5432')
        
        self.connection_string = f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self.engine = None
        
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a SELECT query and return DataFrame"""
        try:
            with self.engine.connect() as conn:
                if params:
                    df = pd.read_sql(text(query), conn, params=params)
                else:
                    df = pd.read_sql(text(query), conn)
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            return pd.DataFrame()
    
    def execute_non_query(self, query: str, params: Optional[Dict] = None) -> bool:
        """Execute INSERT/UPDATE/DELETE query"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(query), params or {})
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Non-query execution failed: {str(e)}")
            return False
    
    def store_train_data(self, df: pd.DataFrame) -> bool:
        if df.empty:
            logger.warning("No train data to store")
            return False
        
        try:
            df_clean = df.copy()
            
            required_columns = ['train_id', 'timestamp', 'station_code']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Insert data in batches
            batch_size = 100
            total_records = len(df_clean)
            
            for i in range(0, total_records, batch_size):
                batch = df_clean.iloc[i:i+batch_size]
                # Map columns to match database schema
                schema_columns = {
                    'train_id': 'train_id',
                    'timestamp': 'timestamp', 
                    'station_code': 'station_code',
                    'scheduled_time': 'scheduled_time',
                    'actual_time': 'actual_time',
                    'delay_minutes': 'delay_minutes',
                    'train_category': 'train_category',
                    'route': 'route',
                    'delay_status': 'delay_status',
                    'destination': 'destination',
                    'is_cancelled': 'is_cancelled'
                }
                
                # Select only columns that exist in the schema
                batch_mapped = batch.copy()
                for col in list(batch_mapped.columns):
                    if col not in schema_columns:
                        batch_mapped = batch_mapped.drop(columns=[col])
                
                batch_mapped.to_sql(
                    'trains',  # Correct table name
                    self.engine, 
                    if_exists='append', 
                    index=False,
                    chunksize=batch_size
                )
            
            logger.info(f"Stored {len(df_clean)} train records in batches")
            return True
            
        except Exception as e:
            logger.error(f"Error storing train data: {str(e)}")
            return False
    
    def store_weather_data(self, df: pd.DataFrame) -> bool:
        """Store weather data in the database"""
        if df.empty:
            logger.warning("No weather data to store")
            return False
        
        try:
            df_clean = df.copy()
            
            # Map CSV columns to DB columns
            column_mapping = {
                'station_code': 'station_code',
                'timestamp': 'timestamp', 
                'temperature': 'temperature',
                'wind_speed': 'wind_speed',
                'precip_mm': 'precip_mm',
                'weather_code': 'weather_code'
            }
            
            # Check required columns
            required_columns = ['station_code', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Select and rename columns
            db_columns = [col for col in column_mapping.keys() if col in df_clean.columns]
            df_clean = df_clean[db_columns].rename(columns=column_mapping)
            
            # Convert timestamp to proper format - handle the specific format from CSV files
            # Use errors='coerce' to handle any parsing issues and then drop invalid rows
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], format='%Y-%m-%dT%H:%M', errors='coerce')
            
            # Remove any rows where timestamp parsing failed (like header rows)
            df_clean = df_clean.dropna(subset=['timestamp'])
            
            # Insert data in batches with duplicate handling
            batch_size = 100
            total_records = len(df_clean)
            inserted_count = 0
            
            for i in range(0, total_records, batch_size):
                batch = df_clean.iloc[i:i+batch_size]
                
                try:
                    # Use pandas to_sql - duplicates will be handled by UNIQUE constraint
                    batch.to_sql(
                        'weather', 
                        self.engine, 
                        if_exists='append', 
                        index=False,
                        method='multi',
                        chunksize=batch_size
                    )
                    inserted_count += len(batch)
                except Exception as e:
                    # If batch fails due to duplicates, try individual inserts
                    logger.warning(f"Batch insert failed, trying individual inserts: {e}")
                    for _, row in batch.iterrows():
                        try:
                            row_df = pd.DataFrame([row])
                            row_df.to_sql(
                                'weather', 
                                self.engine, 
                                if_exists='append', 
                                index=False
                            )
                            inserted_count += 1
                        except Exception:
                            # Skip duplicates silently
                            continue
            
            logger.info(f"Stored {inserted_count}/{len(df_clean)} weather records (duplicates skipped)")
            return True
            
        except Exception as e:
            logger.error(f"Error storing weather data: {str(e)}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            return False
     