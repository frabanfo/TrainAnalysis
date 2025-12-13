"""
Data Quality System for Train and Weather Data Pipeline

This module provides comprehensive data quality validation and monitoring
for the train and weather data pipeline, implementing dual-phase validation:
pre-integration and post-integration quality checks.
"""

from .base import (
    ValidationResult,
    RuleResult,
    QualityReport,
    BaseDataQualityProcessor,
    QualityMetricsStore,
    ValidationState,
    RuleType
)
from .models import (
    DeduplicationResult,
    MissingValueResult,
    BusinessRuleResult,
    CoverageResult,
    PreIntegrationValidationResult,
    PreIntegrationQualityReport,
    QualityReportSerializer
)
from .metrics_store import DatabaseQualityMetricsStore
from .unified_processor import UnifiedDataQualityProcessor, create_train_processor, create_weather_processor
from .config import (
    DataQualitySystemConfig,
    TrainDataQualityConfig,
    WeatherDataQualityConfig,
    IntegrationQualityConfig,
    QualityReportConfig,
    DEFAULT_CONFIG
)
# Report generation is integrated in PreIntegrationProcessor
from .pre_integration_processor import PreIntegrationProcessor
# PreIntegrationMetricsStore unified into DatabaseQualityMetricsStore

__all__ = [
    'ValidationResult',
    'RuleResult', 
    'QualityReport',
    'BaseDataQualityProcessor',
    'QualityMetricsStore',
    'DatabaseQualityMetricsStore',
    'UnifiedDataQualityProcessor',
    'create_train_processor',
    'create_weather_processor',
    'ValidationState',
    'RuleType',
    'DeduplicationResult',
    'MissingValueResult',
    'BusinessRuleResult',
    'CoverageResult',
    'PreIntegrationValidationResult',
    'PreIntegrationQualityReport',
    'QualityReportSerializer',
    'PreIntegrationProcessor',
    'DataQualitySystemConfig',
    'TrainDataQualityConfig',
    'WeatherDataQualityConfig',
    'IntegrationQualityConfig',
    'QualityReportConfig',
    'DEFAULT_CONFIG'
]