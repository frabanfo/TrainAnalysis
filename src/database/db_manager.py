"""
Database manager for PostgreSQL operations
"""

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
                batch.to_sql(
                    'train_data', 
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
     