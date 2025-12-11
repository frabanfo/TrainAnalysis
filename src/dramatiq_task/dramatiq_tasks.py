from dramatiq import group, pipeline
from loguru import logger
from .train_tasks import collect_train_data
from .weather_tasks import collect_weather_data

def full_data_pipeline(start_date: str, end_date: str, chunk_id: str = None):

    parallel_collection = group([
        collect_train_data.message(start_date, end_date, f"train_{chunk_id}"),
        collect_weather_data.message(start_date, end_date, f"weather_{chunk_id}")
    ])
    # train_quality = process_data_quality.message("train", start_date, end_date)
    # weather_quality = process_data_quality.message("weather", start_date, end_date)
    # integration = integrate_data.message(start_date, end_date)
    
    return [
        parallel_collection,
        # train_quality,
        # weather_quality,
        # integration
    ]
