import os
from datetime import datetime
from loguru import logger

def setup_dramatiq_logging(worker_type: str = "worker"):
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]  # Include milliseconds
    log_file = f"logs/dramatiq_{worker_type}_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="100 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    logger.info(f"Dramatiq {worker_type} logging started - {log_file}")
    return log_file

def setup_task_logging(task_name: str, chunk_id: str = None):
    os.makedirs("logs", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')[:-3]
    chunk_suffix = f"_{chunk_id}" if chunk_id else ""
    log_file = f"logs/task_{task_name}{chunk_suffix}_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="50 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    
    logger.info(f"Task {task_name} logging started - {log_file}")
    return log_file