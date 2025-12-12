from dramatiq import group, pipeline
from loguru import logger
from .train_tasks import collect_train_data
from .weather_tasks import collect_weather_data
from .integration_tasks import integrate_train_weather_data

def full_data_pipeline(start_date: str, end_date: str, chunk_id: str = None):
    parallel_collection = group([
        collect_train_data.message(start_date, end_date, f"train_{chunk_id}"),
        collect_weather_data.message(start_date, end_date, f"weather_{chunk_id}")
    ])
    
    integration = integrate_train_weather_data.message(start_date, end_date, f"integration_{chunk_id}")
    
    return [
        parallel_collection,
        integration,
    ]
