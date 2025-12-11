"""
Cleanup tasks for Dramatiq queue maintenance.
"""
import dramatiq
import psycopg2
import os
from datetime import datetime, timedelta
from .dramatiq_config import broker

@dramatiq.actor(queue_name="default", store_results=False)
def cleanup_old_processed_messages(hours_old: int = 1):
    # Database connection parameters
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'railway_analysis')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'railway_user')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'railway_pass')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_old)
        
        # Clean up old processed messages and expired results from queue table
        cursor.execute("""
            DELETE FROM dramatiq.queue 
            WHERE (mtime < %s AND state IN ('done')) 
            OR (result_ttl IS NOT NULL AND result_ttl < NOW())
        """, (cutoff_time,))

        # left rejected and failed messages in case of debugging
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"Cleaned up {deleted_count} old messages and expired results from dramatiq.queue")
        
        cursor.close()
        conn.close()
        
        return {"deleted_messages": deleted_count}
        
    except Exception as e:
        print(f"Error cleaning up messages: {e}")
        raise