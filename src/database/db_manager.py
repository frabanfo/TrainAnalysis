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
                df = pd.read_sql(text(query), conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
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
