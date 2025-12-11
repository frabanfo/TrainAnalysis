import os
import sys
import time
from datetime import datetime, timedelta
from loguru import logger
from dramatiq import pipeline
from dramatiq_task.dramatiq_tasks import full_data_pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    logger.add("logs/scheduler_{time}.log", rotation="1 day", retention="30 days")
    logger.info("Dramatiq Scheduler started")

def start_pipeline(start_date: datetime, end_date: datetime):
    chunk_size_days = 5
    current_start = start_date
    pipelines = []
    chunk_id = 0
    
    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=chunk_size_days-1), end_date)
        chunk_id += 1
        
        logger.info(f"Scheduling pipeline chunk {chunk_id}: {current_start.date()} to {current_end.date()}")
        
        # Create pipeline for this chunk
        pipeline_messages = full_data_pipeline(
            current_start.isoformat(),
            current_end.isoformat(),
            f"chunk_{chunk_id}"
        )
        
        # Send each message individually to ensure both train and weather tasks are queued
        for message in pipeline_messages:
            # Use the pipeline approach but run it immediately
            pipeline([message]).run()
            logger.info(f"📤 Queued task: {message.actor_name} for chunk {chunk_id}")
        
        pipelines.append(len(pipeline_messages))
        current_start = current_end + timedelta(days=1)
    
    total_tasks = sum(pipelines)
    logger.info(f"📊 Scheduled {len(pipelines)} pipeline chunks with {total_tasks} total tasks ({chunk_id} chunks)")
    
    return {
        'pipelines': len(pipelines),
        'total_tasks': total_tasks,
        'chunks': chunk_id
    }


if __name__ == "__main__":
    try:
        setup_logging()
        logger.info("Starting Historical Data Collection Scheduler")

        days_back = int(os.getenv('COLLECTION_DAYS', '30'))

        today = datetime.now().date()
        end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
        start_date = end_date - timedelta(days=days_back - 1)
        
        logger.info(f"Collection period: {days_back} days")
        logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
        
        result = start_pipeline(start_date, end_date)

        logger.info(f"Scheduling result: {result}")
        logger.info("Scheduler completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler failed: {str(e)}")
        raise
