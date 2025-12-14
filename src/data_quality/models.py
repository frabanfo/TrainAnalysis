"""
Data models for the data quality system.

This module defines the data structures used throughout the data quality pipeline,
including validation results, quality reports, and serialization utilities.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
import json
from enum import Enum


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    original_count: int
    duplicate_count: int
    final_count: int
    deduplication_strategy: str
    key_fields: List[str]
    processing_time: float


@dataclass
class MissingValueResult:
    """Result of missing value analysis and handling."""
    field_name: str
    missing_count: int
    missing_percentage: float
    handling_strategy: str  # 'drop', 'flag', 'impute', 'none'
    imputed_count: int = 0
    flagged_count: int = 0


@dataclass
class BusinessRuleResult:
    """Result of business rule validation."""
    rule_name: str
    rule_description: str
    violations_count: int
    violations_percentage: float
    action_taken: str  # 'drop', 'flag', 'transform', 'none'
    threshold_value: Optional[Union[int, float]] = None
    examples: List[str] = field(default_factory=list)


@dataclass
class CoverageResult:
    """Result of data coverage analysis."""
    entity_type: str  # 'station', 'time_period', 'train_route'
    total_expected: int
    total_actual: int
    coverage_percentage: float
    missing_entities: List[str] = field(default_factory=list)
    low_coverage_entities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PreIntegrationValidationResult:
    """Comprehensive result for pre-integration data quality validation."""
    
    # Completeness metrics
    completeness_score: float
    missing_values: List[MissingValueResult]
    
    # Accuracy metrics  
    accuracy_score: float
    business_rules: List[BusinessRuleResult]
    
    # Consistency metrics
    consistency_score: float
    timestamp_issues: int
    schema_violations: int
    
    # Uniqueness metrics
    uniqueness_score: float
    deduplication: DeduplicationResult
    
    # Coverage metrics
    coverage_score: float
    coverage_analysis: List[CoverageResult]
    
    # Overall metrics
    overall_quality_score: float
    processing_timestamp: datetime
    records_processed: int
    records_valid: int
    records_flagged: int
    records_dropped: int


@dataclass
class PreIntegrationQualityReport:
    """Complete quality report for pre-integration validation."""
    report_id: str
    processor_type: str  # 'train', 'weather'
    generation_timestamp: datetime
    date_range: tuple[date, date]
    station_codes: List[str]
    
    # Validation results
    validation_result: PreIntegrationValidationResult
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    config_used: Dict[str, Any] = field(default_factory=dict)
    processing_duration: float = 0.0
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics for the report."""
        return {
            'overall_quality_score': self.validation_result.overall_quality_score,
            'completeness_score': self.validation_result.completeness_score,
            'accuracy_score': self.validation_result.accuracy_score,
            'consistency_score': self.validation_result.consistency_score,
            'uniqueness_score': self.validation_result.uniqueness_score,
            'coverage_score': self.validation_result.coverage_score,
            'records_processed': self.validation_result.records_processed,
            'records_valid': self.validation_result.records_valid,
            'records_flagged': self.validation_result.records_flagged,
            'records_dropped': self.validation_result.records_dropped,
            'processing_duration': self.processing_duration
        }


class QualityReportSerializer:
    """Utility class for serializing and deserializing quality reports."""
    
    @staticmethod
    def to_dict(report: PreIntegrationQualityReport) -> Dict[str, Any]:
        """Convert quality report to dictionary."""
        return asdict(report)
    
    @staticmethod
    def to_json(report: PreIntegrationQualityReport) -> str:
        """Convert quality report to JSON string."""
        def json_serializer(obj):
            if isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(asdict(report), default=json_serializer, indent=2)
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PreIntegrationQualityReport:
        """Create quality report from dictionary."""
        # Handle datetime conversion
        if 'generation_timestamp' in data:
            data['generation_timestamp'] = datetime.fromisoformat(data['generation_timestamp'])
        
        if 'date_range' in data and isinstance(data['date_range'], list):
            data['date_range'] = tuple(
                date.fromisoformat(d) if isinstance(d, str) else d 
                for d in data['date_range']
            )
        
        # Handle nested validation result
        if 'validation_result' in data:
            val_data = data['validation_result']
            if 'processing_timestamp' in val_data:
                val_data['processing_timestamp'] = datetime.fromisoformat(val_data['processing_timestamp'])
            
            # Convert nested objects
            if 'missing_values' in val_data:
                val_data['missing_values'] = [
                    MissingValueResult(**mv) for mv in val_data['missing_values']
                ]
            
            if 'business_rules' in val_data:
                val_data['business_rules'] = [
                    BusinessRuleResult(**br) for br in val_data['business_rules']
                ]
            
            if 'deduplication' in val_data:
                val_data['deduplication'] = DeduplicationResult(**val_data['deduplication'])
            
            if 'coverage_analysis' in val_data:
                val_data['coverage_analysis'] = [
                    CoverageResult(**ca) for ca in val_data['coverage_analysis']
                ]
            
            data['validation_result'] = PreIntegrationValidationResult(**val_data)
        
        return PreIntegrationQualityReport(**data)
    
    @staticmethod
    def from_json(json_str: str) -> PreIntegrationQualityReport:
        """Create quality report from JSON string."""
        data = json.loads(json_str)
        return QualityReportSerializer.from_dict(data)