try:
    from .train_tasks import collect_train_data, collect_train_data_with_dq
    from .weather_tasks import collect_weather_data, collect_weather_data_with_dq
    from .integration_tasks import integrate_train_weather_data
except ImportError:
    # Fallback for when running as script
    from train_tasks import collect_train_data, collect_train_data_with_dq
    from weather_tasks import collect_weather_data, collect_weather_data_with_dq
    from integration_tasks import integrate_train_weather_data

def full_data_pipeline(start_date: str, end_date: str, chunk_id: str = None):
    """Original pipeline with data collection and integration."""
    train_result = collect_train_data.send(start_date, end_date, f"train_{chunk_id}")
    weather_result = collect_weather_data.send(start_date, end_date, f"weather_{chunk_id}")
    integration_result = integrate_train_weather_data.send(start_date, end_date, f"integration_{chunk_id}")
    
    return [
        train_result,
        weather_result,
        integration_result,
    ]

def full_data_pipeline_with_dq(start_date: str, end_date: str, chunk_id: str = None):
    """Enhanced pipeline with data quality validation and integration."""
    train_result = collect_train_data_with_dq.send(start_date, end_date, f"train_dq_{chunk_id}")
    weather_result = collect_weather_data_with_dq.send(start_date, end_date, f"weather_dq_{chunk_id}")
    integration_result = integrate_train_weather_data.send(start_date, end_date, f"integration_dq_{chunk_id}")
    
    return [
        train_result,
        weather_result,
        integration_result,
    ]
