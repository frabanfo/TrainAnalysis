"""
Periodic cleanup scheduler for Dramatiq messages.
Runs cleanup every hour to remove old processed messages.
"""
import time
import schedule
import logging
from dramatiq_task.cleanup_tasks import cleanup_old_processed_messages

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_cleanup():
    try:
        logger.info("Starting periodic cleanup of old Dramatiq messages...")
        cleanup_old_processed_messages.send(hours_old=1)
        logger.info("Cleanup task scheduled successfully")
    except Exception as e:
        logger.error(f"Error scheduling cleanup: {e}")

def main():
    logger.info("Starting Dramatiq cleanup scheduler...")
    
    schedule.every().hour.do(run_cleanup)
    
    run_cleanup()
    
    while True:
        schedule.run_pending()
        time.sleep(60) 

if __name__ == "__main__":
    main()