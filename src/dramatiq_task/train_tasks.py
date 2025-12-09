import dramatiq
from .dramatiq_config import TRAIN_QUEUE

@dramatiq.actor(queue_name=TRAIN_QUEUE)
def process_train(train_id: int):
    print(f"[TRAIN] Processing train ID: {train_id}")
    return {"status": "ok", "train_id": train_id}
