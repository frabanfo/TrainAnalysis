"""
Scheduler with Data Quality integration.
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.dramatiq_task.dramatiq_tasks import full_data_pipeline, full_data_pipeline_with_dq

def setup_logging():
    """Setup logging for the scheduler."""
    log_file = f"logs/scheduler_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    logger.add(log_file, rotation="1 day", retention="30 days")
    logger.info("Dramatiq Scheduler started")

def start_pipeline(start_date: datetime, end_date: datetime, enable_dq: bool = True):
    """Start the data collection pipeline."""
    chunk_size_days = 5
    current_start = start_date
    pipelines = []
    chunk_id = 0
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_size_days-1), end_date)
        chunk_id += 1
        
        logger.info(f"Scheduling pipeline chunk {chunk_id}: {current_start.date()} to {current_end.date()}")
        
        try:
            if enable_dq:
                pipeline_results = full_data_pipeline_with_dq(
                    current_start.isoformat(),
                    current_end.isoformat(),
                    f"dq_chunk_{chunk_id}"
                )
            else:
                pipeline_results = full_data_pipeline(
                    current_start.isoformat(),
                    current_end.isoformat(),
                    f"chunk_{chunk_id}"
                )
            
            logger.info(f"Pipeline chunk {chunk_id} scheduled successfully")
            pipelines.append(pipeline_results)
            
        except Exception as e:
            logger.error(f"Failed to create pipeline chunk {chunk_id}: {e}")
            continue
        
        current_start = current_end + timedelta(days=1)
    
    return {
        'pipelines': len(pipelines),
        'chunks': chunk_id,
        'total_tasks': sum(len(p) for p in pipelines),
        'dq_enabled': enable_dq
    }



def main():
    """Main scheduler function."""
    try:
        # Get configuration
        enable_dq = os.getenv('ENABLE_DQ', 'true').lower() == 'true'
        days_back = int(os.getenv('COLLECTION_DAYS', '7'))
        
        setup_logging()
        logger.info(f"Starting Data Collection Scheduler (DQ: {enable_dq})")
        
        # Calculate date range
        today = datetime.now().date()
        end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
        start_date = end_date - timedelta(days=days_back - 1)
        
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        # Start the pipeline
        result = start_pipeline(start_date, end_date, enable_dq)
        
        logger.info(f"Scheduling complete: {result}")
        logger.info("Start workers: dramatiq src.dramatiq_task.dramatiq_config")
        
        return result
        
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
        return None
        
    except Exception as e:
        logger.error(f"Scheduler failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        result = main()
        if result:
            print(f"Pipeline scheduled: {result['total_tasks']} tasks across {result['chunks']} chunks")
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"Scheduler crashed: {e}")
        sys.exit(1)