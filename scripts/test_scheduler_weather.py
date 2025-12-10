#!/usr/bin/env python3
"""
Test per verificare che lo scheduler includa correttamente la weather task.
"""

import os
import sys
from datetime import datetime, timedelta
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dramatiq_task.dramatiq_tasks import full_data_pipeline


def main():
    """Test che lo scheduler includa la weather task"""
    
    logger.info("[TEST SCHEDULER] Testing scheduler weather integration")
    
    # Test con un piccolo range di date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    logger.info(f"[TEST SCHEDULER] Date range: {start_date.date()} → {end_date.date()}")
    
    # Ottieni i messaggi del pipeline
    pipeline_messages = full_data_pipeline(
        start_date.isoformat(),
        end_date.isoformat(),
        "test_scheduler"
    )
    
    logger.info(f"[TEST SCHEDULER] Pipeline contains {len(pipeline_messages)} tasks:")
    
    for i, message in enumerate(pipeline_messages):
        task_name = message.actor_name
        logger.info(f"[TEST SCHEDULER] Task {i+1}: {task_name}")
    
    # Verifica che ci sia la weather task
    weather_tasks = [msg for msg in pipeline_messages if 'weather' in msg.actor_name]
    train_tasks = [msg for msg in pipeline_messages if 'train' in msg.actor_name]
    
    if weather_tasks:
        logger.info(f"[TEST SCHEDULER] ✅ Weather task found: {weather_tasks[0].actor_name}")
    else:
        logger.error("[TEST SCHEDULER] ❌ Weather task NOT found in pipeline!")
    
    if train_tasks:
        logger.info(f"[TEST SCHEDULER] ✅ Train task found: {train_tasks[0].actor_name}")
    else:
        logger.error("[TEST SCHEDULER] ❌ Train task NOT found in pipeline!")
    
    logger.info("[TEST SCHEDULER] Scheduler integration test completed")


if __name__ == "__main__":
    main()