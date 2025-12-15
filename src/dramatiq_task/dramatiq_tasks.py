from .train_tasks import collect_train_data
from .weather_tasks import collect_weather_data
from .integration_tasks import integrate_train_weather_data
import dramatiq

@dramatiq.actor(queue_name="default", max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def full_data_pipeline(start_date: str, end_date: str, chunk_id: str):
    train_msg = collect_train_data.message(start_date, end_date, f"train_dq_{chunk_id}")
    weather_msg = collect_weather_data.message(start_date, end_date, f"weather_dq_{chunk_id}")
    intergation_msg = integrate_train_weather_data.message(start_date, end_date, f"integration_dq_{chunk_id}")
    group = dramatiq.group([train_msg, weather_msg])
    group.add_completion_callback(intergation_msg)
    group.run()
    return f"Pipeline initiated for chunk {chunk_id} from {start_date} to {end_date}"
