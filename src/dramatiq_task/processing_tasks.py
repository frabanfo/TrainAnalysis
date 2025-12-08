import dramatiq
from .dramatiq_config import PROCESSING_QUEUE

@dramatiq.actor(queue_name=PROCESSING_QUEUE)
def process_data(data: dict):
    print(f"[PROCESS] Running data pipeline on: {data}")
    return {"processed": True, "input": data}
