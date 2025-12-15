import os
import dramatiq
from dramatiq_pg.broker import PostgresBroker
from dramatiq_pg.results import PostgresBackend
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import AgeLimit, TimeLimit, Callbacks, Pipelines, Retries, GroupCallbacks
from dramatiq.results import Results
from dramatiq.rate_limits.backends.redis import RedisBackend as RedisBarrierBackend

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'railway_analysis')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'railway_user')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'railway_pass')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}?minconn=5&maxconn=20"

broker = PostgresBroker(url=DATABASE_URL)

results_backend = PostgresBackend(url=DATABASE_URL)


# Proper backends
barrier_backend = RedisBarrierBackend(url=REDIS_URL)   # for group completion tracking

broker.add_middleware(AgeLimit(max_age=3600000))
broker.add_middleware(TimeLimit(time_limit=1800000))
broker.add_middleware(Callbacks())
broker.add_middleware(Pipelines())
broker.add_middleware(Retries(max_retries=3))
broker.add_middleware(Results(backend=results_backend))
broker.add_middleware(GroupCallbacks(barrier_backend))

dramatiq.set_broker(broker)

# Queues
TRAIN_QUEUE = "train_queue"
WEATHER_QUEUE = "weather_queue" 
PROCESSING_QUEUE = "processing_queue"
DEFAULT_QUEUE = "default"
