try:
    from .train_tasks import collect_train_data, collect_train_data_with_dq
    from .weather_tasks import collect_weather_data, collect_weather_data_with_dq
except ImportError:
    # Fallback for when running as script
    from train_tasks import collect_train_data, collect_train_data_with_dq
    from weather_tasks import collect_weather_data, collect_weather_data_with_dq

def full_data_pipeline(start_date: str, end_date: str, chunk_id: str = None):
    """Original pipeline without data quality validation."""
    # Send tasks directly instead of creating message objects
    train_result = collect_train_data.send(start_date, end_date, f"train_{chunk_id}")
    weather_result = collect_weather_data.send(start_date, end_date, f"weather_{chunk_id}")
    
    return [
        train_result,
        weather_result,
    ]

def full_data_pipeline_with_dq(start_date: str, end_date: str, chunk_id: str = None):
    """Enhanced pipeline with integrated data quality validation."""
    # Send tasks directly instead of creating message objects
    train_result = collect_train_data_with_dq.send(start_date, end_date, f"train_dq_{chunk_id}")
    weather_result = collect_weather_data_with_dq.send(start_date, end_date, f"weather_dq_{chunk_id}")
    
    return [
        train_result,
        weather_result,
    ]
