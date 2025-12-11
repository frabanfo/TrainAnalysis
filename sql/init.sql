-- Create Dramatiq schema and tables
CREATE SCHEMA IF NOT EXISTS dramatiq;

-- Dramatiq queue table with correct schema
CREATE TABLE IF NOT EXISTS dramatiq.queue (
    id BIGSERIAL PRIMARY KEY,
    queue_name TEXT NOT NULL,
    message_id TEXT NOT NULL UNIQUE,
    state TEXT NOT NULL DEFAULT 'queued',
    message JSONB NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    mtime TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Dramatiq results table
CREATE TABLE IF NOT EXISTS dramatiq.results (
    message_id TEXT PRIMARY KEY,
    result BYTEA,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ttl TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_dramatiq_queue_state ON dramatiq.queue(state);
CREATE INDEX IF NOT EXISTS idx_dramatiq_queue_priority ON dramatiq.queue(priority DESC, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_dramatiq_queue_name ON dramatiq.queue(queue_name);
CREATE INDEX IF NOT EXISTS idx_dramatiq_queue_mtime ON dramatiq.queue(mtime);
CREATE INDEX IF NOT EXISTS idx_dramatiq_results_ttl ON dramatiq.results(ttl);

-- Application tables
CREATE TABLE IF NOT EXISTS stations (
    station_code TEXT PRIMARY KEY,
    station_name TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trains (
    id SERIAL PRIMARY KEY,
    train_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    station_code TEXT REFERENCES stations(station_code),
    scheduled_time TIMESTAMP WITH TIME ZONE,
    actual_time TIMESTAMP WITH TIME ZONE,
    delay_minutes INTEGER,
    train_category TEXT,
    route TEXT,
    delay_status TEXT CHECK (delay_status IN ('on_time', 'delayed', 'early', 'cancelled')),
    destination TEXT,
    is_cancelled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(train_id, timestamp, station_code)
);

CREATE TABLE IF NOT EXISTS weather (
    id SERIAL PRIMARY KEY,
    station_code TEXT REFERENCES stations(station_code),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    temperature REAL,
    wind_speed REAL,
    precip_mm REAL,
    weather_code INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(station_code, timestamp)
);

CREATE TABLE IF NOT EXISTS train_weather_integrated (
    id SERIAL PRIMARY KEY,
    train_id TEXT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    station_code TEXT REFERENCES stations(station_code),
    delay_minutes INTEGER,
    temperature REAL,
    wind_speed REAL,
    precip_mm REAL,
    weather_code INTEGER,
    train_category TEXT,
    route TEXT,
    delay_status TEXT CHECK (delay_status IN ('on_time', 'delayed', 'early', 'cancelled')),
    destination TEXT,
    is_cancelled BOOLEAN DEFAULT FALSE,
    hour_of_day INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN,
    is_rush_hour BOOLEAN,
    temp_category TEXT,
    is_raining BOOLEAN,
    rain_intensity TEXT,
    wind_category TEXT,
    is_delayed BOOLEAN,
    delay_category TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS data_quality_metrics (
    id SERIAL PRIMARY KEY,
    table_name TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- Essential indexes only
CREATE INDEX IF NOT EXISTS idx_trains_station_timestamp ON trains(station_code, timestamp);
CREATE INDEX IF NOT EXISTS idx_weather_station_timestamp ON weather(station_code, timestamp);
CREATE INDEX IF NOT EXISTS idx_integrated_station_timestamp ON train_weather_integrated(station_code, timestamp);
CREATE INDEX IF NOT EXISTS idx_trains_delay_status ON trains(delay_status);
CREATE INDEX IF NOT EXISTS idx_integrated_delay_status ON train_weather_integrated(delay_status);