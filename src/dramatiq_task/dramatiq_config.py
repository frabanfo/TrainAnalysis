import os
import dramatiq
from dramatiq_pg.broker import PostgresBroker
from dramatiq_pg.results import PostgresBackend
from dramatiq.middleware import AgeLimit, TimeLimit, Callbacks, Pipelines, Retries
from dramatiq.results import Results

POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'railway_analysis')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'railway_user')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'railway_pass')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

broker = PostgresBroker(url=DATABASE_URL)

results_backend = PostgresBackend(url=DATABASE_URL)
broker.add_middleware(AgeLimit(max_age=3600000))  # 1 hour max age - auto cleanup old messages
broker.add_middleware(TimeLimit(time_limit=1800000))  # 30 min time limit
broker.add_middleware(Callbacks())
broker.add_middleware(Pipelines())
broker.add_middleware(Retries(max_retries=3))
broker.add_middleware(Results(backend=results_backend))

dramatiq.set_broker(broker)

TRAIN_QUEUE = "train_queue"
WEATHER_QUEUE = "weather_queue" 
PROCESSING_QUEUE = "processing_queue"
DEFAULT_QUEUE = "default"