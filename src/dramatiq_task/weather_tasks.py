import dramatiq
from .dramatiq_config import WEATHER_QUEUE

@dramatiq.actor(queue_name=WEATHER_QUEUE)
def fetch_weather(station_code: str):
    print(f"[WEATHER] Fetching weather for station: {station_code}")
    return {"station": station_code, "weather": "sunny"}
