"""
Base interfaces and abstract classes for data quality processors.

This module defines the core abstractions used throughout the data quality system,
including validation results, quality reports, and the base processor interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd


class ValidationState(Enum):
    """Enumeration of validation states for records."""
    VALID = "valid"
    INVALID = "invalid"
    FLAGGED = "flagged"
    DROPPED = "dropped"


class RuleType(Enum):
    """Enumeration of data quality rule types."""
    SCHEMA = "schema"
    BUSINESS = "business"
    RANGE = "range"
    CONSISTENCY = "consistency"
    COMPLETENESS = "completeness"
    DEDUPLICATION = "deduplication"


@dataclass
class RuleResult:
    """Result of applying a single data quality rule."""
    rule_name: str
    rule_type: RuleType
    passed: bool
    affected_records: int
    violation_details: List[str] = field(default_factory=list)
    action_taken: str = "none"  # 'drop', 'flag', 'transform', 'none'


@dataclass
class ValidationResult:
    """Result of data quality validation for a dataset."""
    processor_type: str  # 'train', 'weather', 'integration'
    timestamp: datetime
    station_code: Optional[str] = None
    total_records: int = 0
    valid_records: int = 0
    dropped_records: int = 0
    flagged_records: int = 0
    validation_rules: List[RuleResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    anomalies: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality report for a data processing run."""
    report_id: str
    report_type: str  # 'train', 'weather', 'integration'
    generation_timestamp: datetime
    date_range: Tuple[date, date]
    station_codes: List[str]
    summary_metrics: Dict[str, float] = field(default_factory=dict)
    detailed_results: List[ValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv'])


class BaseDataQualityProcessor(ABC):
    """
    Abstract base class for all data quality processors.
    
    This class defines the common interface that all DQ processors must implement,
    ensuring consistent behavior across train, weather, and integration processors.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality processor.
        
        Args:
            config: Optional configuration dictionary for processor settings
        """
        self.config = config or {}
        self.validation_results: List[ValidationResult] = []
    
    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the schema and required fields of the dataset.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            ValidationResult containing schema validation details
        """
        pass
    
    @abstractmethod
    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """
        Remove duplicate records from the dataset.
        
        Args:
            df: Input DataFrame with potential duplicates
            
        Returns:
            Tuple of (cleaned DataFrame, ValidationResult)
        """
        pass
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[ValidationResult]]:
        """
        Execute the complete data quality processing pipeline.
        
        Args:
            df: Input DataFrame to process
            
        Returns:
            Tuple of (processed DataFrame, list of ValidationResults)
        """
        pass
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation results.
        
        Returns:
            Dictionary containing validation summary statistics
        """
        if not self.validation_results:
            return {}
        
        total_records = sum(result.total_records for result in self.validation_results)
        total_valid = sum(result.valid_records for result in self.validation_results)
        total_dropped = sum(result.dropped_records for result in self.validation_results)
        total_flagged = sum(result.flagged_records for result in self.validation_results)
        
        return {
            'total_records_processed': total_records,
            'total_valid_records': total_valid,
            'total_dropped_records': total_dropped,
            'total_flagged_records': total_flagged,
            'validation_success_rate': total_valid / total_records if total_records > 0 else 0.0,
            'drop_rate': total_dropped / total_records if total_records > 0 else 0.0,
            'flag_rate': total_flagged / total_records if total_records > 0 else 0.0
        }
    
    def clear_results(self):
        """Clear all stored validation results."""
        self.validation_results.clear()


class QualityMetricsStore(ABC):
    """
    Abstract interface for storing and retrieving quality metrics.
    
    This interface allows different storage backends (database, file system, etc.)
    to be used for persisting quality metrics and reports.
    """
    
    @abstractmethod
    def store_validation_result(self, result: ValidationResult) -> bool:
        """
        Store a validation result.
        
        Args:
            result: ValidationResult to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def store_quality_report(self, report: QualityReport) -> bool:
        """
        Store a quality report.
        
        Args:
            report: QualityReport to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_reports(self, 
                        report_type: Optional[str] = None,
                        start_date: Optional[date] = None,
                        end_date: Optional[date] = None,
                        station_codes: Optional[List[str]] = None) -> List[QualityReport]:
        """
        Retrieve quality reports based on filters.
        
        Args:
            report_type: Optional filter by report type
            start_date: Optional start date filter
            end_date: Optional end date filter
            station_codes: Optional list of station codes to filter by
            
        Returns:
            List of matching QualityReport objects
        """
        pass