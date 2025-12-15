# TrainAnalysis - Railway Weather Correlation System

A comprehensive data pipeline system for analyzing correlations between train delays and weather conditions in the Lombardy region of Italy. This project implements a robust, scalable architecture for collecting, processing, and integrating railway and meteorological data.

## Software Architecture

### System Overview

The TrainAnalysis system is built using a **microservices architecture** with **event-driven processing** using Dramatiq task queues. The system follows a **data lakehouse pattern** with structured data storage and comprehensive data quality management.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Task Queues    │    │   Storage       │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ TrainStats API  │───▶│ Train Queue     │───▶│ PostgreSQL      │
│ OpenMeteo API   │───▶│ Weather Queue   │───▶│ Raw Data Files  │
│ Station Registry│───▶│ Processing Queue│───▶│ Quality Metrics │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

#### 1. **Data Ingestion Layer**
- **TrainStats Client**: Collects real-time and archive train data from Italian railway APIs
- **OpenMeteo Client**: Fetches historical and current weather data
- **Station Registry**: Manages railway station metadata with coordinates

#### 2. **Task Processing Layer (Dramatiq)**
- **Train Tasks**: Parallel collection of railway data with retry mechanisms
- **Weather Tasks**: Meteorological data collection with geographic matching
- **Integration Tasks**: Temporal and spatial data fusion
- **Scheduler**: Orchestrates data collection workflows

#### 3. **Data Quality Framework**
- **Unified Processor**: Configurable validation for both train and weather data
- **Schema Validation**: Ensures data structure integrity
- **Business Rules**: Domain-specific validation (delay thresholds, weather ranges)
- **Deduplication**: Intelligent duplicate removal with completeness scoring
- **Metrics Store**: Tracks data quality KPIs and validation results

#### 4. **Data Integration Engine**
- **Temporal Matching**: Aligns train events with weather conditions (±60min window)
- **Spatial Matching**: Links weather data to railway stations by coordinates
- **Feature Engineering**: Creates derived features for analysis
- **Quality Scoring**: Calculates integration success metrics

#### 5. **Storage Layer**
- **PostgreSQL**: Primary data warehouse with optimized schemas
- **File System**: Raw data backup and intermediate processing files
- **Redis**: Task queue backend and caching layer

## Data Flow Pipeline

### Phase 1: Data Collection
```
Scheduler ──┐
            ├─▶ Train Collection Task ──┐
            └─▶ Weather Collection Task ─┤
                                         ├─▶ Integration Task ──▶ Storage
                                         │
                                         └─▶ Data Quality Validation
```

### Phase 2: Processing Workflow

1. **Initialization**
   - Scheduler calculates date ranges based on `COLLECTION_DAYS` environment variable
   - Creates processing chunks to manage memory and enable parallel processing
   - Dispatches tasks to appropriate queues

2. **Parallel Data Collection**
   - **Train Data**: Fetches from TrainStats API for all Lombardy stations
   - **Weather Data**: Retrieves meteorological data from OpenMeteo API
   - Both processes include automatic retry logic and error handling

3. **Data Quality Processing**
   - **Schema Validation**: Checks required fields and data types
   - **Timestamp Normalization**: Converts to UTC and rounds weather data to hourly
   - **Deduplication**: Removes duplicates using configurable key fields
   - **Business Rules**: Applies domain-specific validation rules
   - **Quality Metrics**: Tracks validation results and data quality KPIs

4. **Data Integration**
   - **Temporal Alignment**: Matches train events with weather within 60-minute windows
   - **Spatial Matching**: Links data by station coordinates
   - **Feature Engineering**: Creates analytical features (rush hour, weather categories, etc.)
   - **Quality Assessment**: Calculates match rates and integration success metrics

5. **Storage and Monitoring**
   - Stores processed data in PostgreSQL with proper indexing
   - Maintains data quality metrics for monitoring
   - Logs comprehensive processing statistics

### Data Quality Framework

The system implements a comprehensive data quality framework:

#### Train Data Validation
- **Required Fields**: train_id, timestamp, station_code
- **Delay Validation**: Flags extreme delays (>300min) and impossible early arrivals (<-5min)
- **Deduplication**: Keeps records with highest completeness scores
- **Temporal Validation**: Ensures timestamps are within reasonable ranges

#### Weather Data Validation
- **Required Fields**: station_code, timestamp
- **Range Validation**: Temperature (-50°C to +60°C), precipitation (≥0mm)
- **Temporal Rounding**: Rounds to nearest hour for consistency
- **Deduplication**: Keeps most recent records for same station/time

#### Integration Quality
- **Match Rate**: Percentage of train records successfully matched with weather
- **Temporal Accuracy**: Time difference between train and weather observations
- **Completeness**: Percentage of required fields populated
- **Consistency**: Cross-validation between related fields

## 🚀 Build Instructions

### Prerequisites

- **Docker & Docker Compose**: Container orchestration
- **Git**: Version control
- **8GB+ RAM**: For processing large datasets
- **10GB+ Disk Space**: For data storage and logs

### Quick Start

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd TrainAnalysis
   ```

2. **Environment Setup**
   ```bash
   # Copy environment template (if exists)
   cp .env.example .env  # Edit as needed
   
   # Or set environment variables directly in docker-compose.yml
   # Key variables:
   # - COLLECTION_DAYS: Number of days to collect (default: 30)
   # - CHUNK_SIZE: Processing chunk size in days (default: 1)
   ```

3. **Build and Start Services**
   ```bash
   # Build all containers
   docker-compose build
   
   # Start the complete system
   docker-compose up -d
   
   # Monitor logs
   docker-compose logs -f
   ```

4. **Verify Installation**
   ```bash
   # Check service status
   docker-compose ps
   
   # Verify database connection
   docker-compose exec postgres psql -U railway_user -d railway_analysis -c "\dt"
   
   # Check station initialization
   docker-compose logs stations-init
   
   # Monitor task processing
   docker-compose logs dramatiq-worker
   ```

### Service Architecture

The system runs the following services:

#### Core Services
- **postgres**: PostgreSQL 15 database with optimized configuration
- **redis**: Redis 7 for task queue management
- **stations-init**: One-time station data initialization
- **dramatiq-worker**: Multi-process task worker with 4 queues
- **railway-scheduler**: Main orchestration service

#### Service Dependencies
```
postgres (healthy) ──┐
                     ├─▶ stations-init ──┐
redis (healthy) ─────┤                   ├─▶ railway-scheduler
                     └─▶ dramatiq-worker ─┘
```

### Configuration Options

#### Environment Variables
```bash
# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_DB=railway_analysis
POSTGRES_USER=railway_user
POSTGRES_PASSWORD=railway_pass

# Processing Configuration
COLLECTION_DAYS=30        # Days of data to collect
CHUNK_SIZE=1             # Processing chunk size (days)

# Queue Configuration
REDIS_URL=redis://redis:6379
```

#### Docker Compose Customization
```yaml
# Adjust worker processes and threads
dramatiq-worker:
  command: >
    dramatiq src.dramatiq_task.train_tasks src.dramatiq_task.weather_tasks 
    src.dramatiq_task.integration_tasks src.dramatiq_task.dramatiq_tasks
    --queues train_queue weather_queue processing_queue default
    --processes 4    # Adjust based on CPU cores
    --threads 8      # Adjust based on memory
    --verbose
```

### Development Setup

1. **Local Development Environment**
   ```bash
   # Create Python virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dramatiq.txt
   ```

2. **Database Setup for Development**
   ```bash
   # Start only database services
   docker-compose up -d postgres redis
   
   # Run initialization
   python scripts/init_stations.py
   ```

3. **Manual Task Execution**
   ```bash
   # Start dramatiq worker locally
   dramatiq src.dramatiq_task.train_tasks src.dramatiq_task.weather_tasks \
            src.dramatiq_task.integration_tasks src.dramatiq_task.dramatiq_tasks \
            --processes 1 --threads 2
   
   # Run scheduler manually
   python src/scheduler.py
   ```

### Monitoring and Maintenance

#### Log Management
```bash
# View real-time logs
docker-compose logs -f railway-scheduler
docker-compose logs -f dramatiq-worker

# Check specific service logs
docker-compose logs stations-init
docker-compose logs postgres

# Log files are stored in ./logs/ directory
ls -la logs/
```

#### Database Monitoring
```bash
# Connect to database
docker-compose exec postgres psql -U railway_user -d railway_analysis

# Check data counts
SELECT 
  (SELECT COUNT(*) FROM stations) as stations,
  (SELECT COUNT(*) FROM trains) as trains,
  (SELECT COUNT(*) FROM weather) as weather,
  (SELECT COUNT(*) FROM train_weather_integrated) as integrated;

# Check recent data
SELECT DATE(timestamp), COUNT(*) 
FROM trains 
GROUP BY DATE(timestamp) 
ORDER BY DATE(timestamp) DESC 
LIMIT 10;
```

#### Performance Tuning
```bash
# Adjust PostgreSQL settings in docker-compose.yml
postgres:
  command: postgres -c max_connections=200 -c shared_buffers=512MB -c effective_cache_size=2GB

# Scale worker processes
docker-compose up -d --scale dramatiq-worker=3
```

### Troubleshooting

#### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check PostgreSQL health
   docker-compose exec postgres pg_isready -U railway_user
   
   # Restart database
   docker-compose restart postgres
   ```

2. **Task Queue Issues**
   ```bash
   # Check Redis connectivity
   docker-compose exec redis redis-cli ping
   
   # Clear task queues
   docker-compose exec redis redis-cli FLUSHALL
   ```

3. **Memory Issues**
   ```bash
   # Monitor container memory usage
   docker stats
   
   # Reduce chunk size and worker processes
   # Edit CHUNK_SIZE and dramatiq worker configuration
   ```

4. **Data Quality Issues**
   ```bash
   # Check data quality metrics
   docker-compose exec postgres psql -U railway_user -d railway_analysis \
     -c "SELECT * FROM data_quality_metrics ORDER BY timestamp DESC LIMIT 20;"
   ```

### Production Deployment

#### Security Considerations
- Change default database passwords
- Use environment files for sensitive configuration
- Implement network security groups
- Enable SSL/TLS for database connections
- Set up log rotation and monitoring

#### Scaling Recommendations
- Use managed PostgreSQL service for production
- Implement Redis clustering for high availability
- Deploy multiple worker instances across nodes
- Set up monitoring with Grafana/Prometheus
- Implement automated backup strategies

#### Backup Strategy
```bash
# Database backup
docker-compose exec postgres pg_dump -U railway_user railway_analysis > backup.sql

# Data directory backup
tar -czf data_backup.tar.gz data/ logs/
```

This system provides a robust foundation for analyzing train-weather correlations with enterprise-grade data quality, monitoring, and scalability features.