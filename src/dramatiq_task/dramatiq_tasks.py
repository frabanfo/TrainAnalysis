from .train_tasks import collect_train_data
from .weather_tasks import collect_weather_data
from .integration_tasks import integrate_train_weather_data

def full_data_pipeline(start_date: str, end_date: str, chunk_id: str = None):
    train_result = collect_train_data.send(start_date, end_date, f"train_dq_{chunk_id}")
    weather_result = collect_weather_data.send(start_date, end_date, f"weather_dq_{chunk_id}")
    integration_result = integrate_train_weather_data.send(start_date, end_date, f"integration_dq_{chunk_id}")
    
    return [
        train_result,
        weather_result,
        integration_result,
    ]
