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
        
        cursor.execute("""
            DELETE FROM dramatiq.queue 
            WHERE created_at < %s 
            AND state IN ('done')
        """, (cutoff_time,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"Cleaned up {deleted_count} old messages from dramatiq.queue")
        
        cursor.execute("""
            DELETE FROM dramatiq.results 
            WHERE created_at < %s OR ttl < NOW()
        """, (cutoff_time,))
        
        deleted_results = cursor.rowcount
        conn.commit()
        
        print(f"Cleaned up {deleted_results} old results from dramatiq.results")
        
        cursor.close()
        conn.close()
        
        return {"deleted_messages": deleted_count, "deleted_results": deleted_results}
        
    except Exception as e:
        print(f"Error cleaning up messages: {e}")
        raise