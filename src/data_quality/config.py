"""
Configuration classes for the data quality system.

This module defines configuration classes that control the behavior
of data quality processors and validation rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TrainDataQualityConfig:
    """Configuration for train data quality processing."""
    
    # Schema validation
    required_fields: List[str] = field(default_factory=lambda: [
        'station_code', 'timestamp', 'train_id'
    ])
    
    # Timestamp handling
    target_timezone: str = 'Europe/Rome'
    
    # Deduplication
    deduplication_key_fields: List[str] = field(default_factory=lambda: [
        'train_id', 'timestamp', 'station_code'
    ])
    
    # Missing values
    delay_unknown_value: int = -1
    missing_value_threshold: float = 0.5  # Drop if >50% critical fields missing
    critical_fields: List[str] = field(default_factory=lambda: [
        'station_code', 'timestamp', 'train_id', 'scheduled_time'
    ])
    
    # Business rules - Accuracy
    delay_min_threshold: int = -5  # Minutes - extreme early arrivals
    delay_max_threshold: int = 300  # Minutes - extreme delays (5 hours)
    temperature_min: float = -20.0  # Celsius
    temperature_max: float = 50.0   # Celsius
    
    # Coverage analysis
    min_coverage_threshold: float = 0.8  # 80% minimum coverage expected


@dataclass  
class WeatherDataQualityConfig:
    """Configuration for weather data quality processing."""
    
    # Schema validation
    required_fields: List[str] = field(default_factory=lambda: [
        'station_code', 'timestamp', 'temperature'
    ])
    
    # Timestamp handling
    target_timezone: str = 'Europe/Rome'
    weather_timestamp_tolerance: int = 3600  # 1 hour in seconds
    
    # Deduplication
    deduplication_key_fields: List[str] = field(default_factory=lambda: [
        'station_code', 'timestamp'
    ])
    
    # Business rules - Accuracy
    temperature_min: float = -20.0
    temperature_max: float = 50.0
    temperature_range: tuple = (-20.0, 50.0)  # For compatibility
    precip_mm_min: float = 0.0        # Precipitation must be >= 0
    precip_mm_max: float = 500.0      # mm per hour
    wind_speed_min: float = 0.0       # Wind speed must be >= 0
    wind_speed_max: float = 200.0     # km/h
    weather_code_min: int = 0         # Weather codes are typically >= 0
    weather_code_max: int = 99        # Weather codes are typically <= 99
    
    # Temporal continuity
    max_gap_hours: int = 6            # Maximum gap between weather records
    expected_frequency_hours: int = 1  # Expected hourly data
    
    # Missing values
    missing_value_threshold: float = 0.3  # More lenient for weather
    critical_fields: List[str] = field(default_factory=lambda: [
        'station_code', 'timestamp'
    ])


@dataclass
class IntegrationQualityConfig:
    """Configuration for integration quality checks."""
    
    # Temporal alignment
    max_time_difference: int = 3600  # 1 hour in seconds
    
    # Station matching
    require_station_coordinates: bool = True
    coordinate_precision: int = 6  # Decimal places
    
    # Data completeness for integration
    min_weather_coverage: float = 0.7  # 70% weather data coverage required
    min_train_coverage: float = 0.8    # 80% train data coverage required


@dataclass
class QualityReportConfig:
    """Configuration for quality report generation."""
    
    # Report settings
    include_recommendations: bool = True
    max_examples_per_rule: int = 5
    
    # Export formats
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv'])
    
    # Thresholds for recommendations
    critical_quality_threshold: float = 0.7   # Below this triggers critical recommendations
    warning_quality_threshold: float = 0.85   # Below this triggers warnings


@dataclass
class DataQualitySystemConfig:
    """Main configuration class combining all DQ configurations."""
    
    train_config: TrainDataQualityConfig = field(default_factory=TrainDataQualityConfig)
    weather_config: WeatherDataQualityConfig = field(default_factory=WeatherDataQualityConfig)
    integration_config: IntegrationQualityConfig = field(default_factory=IntegrationQualityConfig)
    report_config: QualityReportConfig = field(default_factory=QualityReportConfig)
    
    # Global settings
    enable_logging: bool = True
    log_level: str = 'INFO'
    parallel_processing: bool = False
    max_workers: int = 4


# Default configuration instance
DEFAULT_CONFIG = DataQualitySystemConfig()