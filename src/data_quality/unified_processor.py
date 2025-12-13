"""
Unified data quality processor for both train and weather data.

This module combines the functionality of TrainDataQualityProcessor and 
WeatherDataQualityProcessor into a single, configurable processor.
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Literal
from loguru import logger
from dateutil import parser

from .base import BaseDataQualityProcessor, ValidationResult, RuleResult, RuleType
from .config import TrainDataQualityConfig, WeatherDataQualityConfig


class UnifiedDataQualityProcessor(BaseDataQualityProcessor):
    """
    Unified data quality processor for both train and weather data.
    
    This processor can handle both data types using a single interface,
    reducing code duplication and simplifying the architecture.
    """
    
    def __init__(self, 
                 data_type: Literal['train', 'weather'],
                 train_config: Optional[TrainDataQualityConfig] = None,
                 weather_config: Optional[WeatherDataQualityConfig] = None):
        """
        Initialize unified data quality processor.
        
        Args:
            data_type: Type of data to process ('train' or 'weather')
            train_config: Configuration for train data processing
            weather_config: Configuration for weather data processing
        """
        super().__init__(None)
        self.data_type = data_type
        self.train_config = train_config or TrainDataQualityConfig()
        self.weather_config = weather_config or WeatherDataQualityConfig()
        
        # Set active config based on data type
        if data_type == 'train':
            self.config = self.train_config
            self.required_fields = self.train_config.required_fields
            self.dedup_key_fields = self.train_config.deduplication_key_fields
        elif data_type == 'weather':
            self.config = self.weather_config
            self.required_fields = self.weather_config.required_fields
            self.dedup_key_fields = self.weather_config.deduplication_key_fields
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'train' or 'weather'")
    
    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate schema for both train and weather data."""
        start_time = datetime.now()
        total_records = len(df)
        
        rules = []
        dropped_count = 0
        
        # Check required fields
        for field in self.required_fields:
            if field not in df.columns:
                rules.append(RuleResult(
                    rule_name=f"{field}_exists",
                    rule_type=RuleType.SCHEMA,
                    passed=False,
                    affected_records=total_records,
                    violation_details=[f"Required {self.data_type} field {field} missing from dataset"],
                    action_taken="drop"
                ))
                dropped_count = total_records
                continue
            
            null_count = df[field].isnull().sum()
            rules.append(RuleResult(
                rule_name=f"{field}_not_null",
                rule_type=RuleType.SCHEMA,
                passed=null_count == 0,
                affected_records=null_count,
                violation_details=[f"{null_count} records have null {field}"] if null_count > 0 else [],
                action_taken="drop" if null_count > 0 else "none"
            ))
            
            if null_count > 0:
                dropped_count = max(dropped_count, null_count)
        
        # Calculate records with any required field null
        if all(field in df.columns for field in self.required_fields):
            null_mask = df[self.required_fields].isnull().any(axis=1)
            dropped_count = null_mask.sum()
        
        valid_records = total_records - dropped_count
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type=self.data_type,
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=valid_records,
            dropped_records=dropped_count,
            flagged_records=0,
            validation_rules=rules,
            metrics={
                'schema_validation_duration': duration,
                'required_fields_count': len(self.required_fields),
                'schema_compliance_rate': valid_records / total_records if total_records > 0 else 0.0
            }
        )
    
    def normalize_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Normalize timestamps for both train and weather data."""
        start_time = datetime.now()
        df_clean = df.copy()
        total_records = len(df_clean)
        
        parsing_errors = 0
        normalized_count = 0
        
        if 'timestamp' in df_clean.columns:
            target_tz = pytz.timezone(self.config.target_timezone)
            
            def normalize_timestamp(ts):
                nonlocal parsing_errors, normalized_count
                try:
                    if pd.isna(ts):
                        return None
                    
                    # Parse timestamp
                    if isinstance(ts, str):
                        dt = parser.parse(ts)
                    elif isinstance(ts, datetime):
                        dt = ts
                    else:
                        parsing_errors += 1
                        return None
                    
                    # Handle timezone
                    if dt.tzinfo is None:
                        dt = pytz.UTC.localize(dt)
                    
                    # Convert to target timezone
                    normalized_dt = dt.astimezone(target_tz)
                    
                    # Round to nearest hour for weather data
                    if self.data_type == 'weather':
                        if normalized_dt.minute >= 30:
                            normalized_dt = normalized_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                        else:
                            normalized_dt = normalized_dt.replace(minute=0, second=0, microsecond=0)
                    
                    normalized_count += 1
                    return normalized_dt
                    
                except (ValueError, TypeError, parser.ParserError):
                    parsing_errors += 1
                    return None
            
            df_clean['timestamp'] = df_clean['timestamp'].apply(normalize_timestamp)
            
            # Drop records with failed timestamp parsing
            initial_count = len(df_clean)
            df_clean = df_clean.dropna(subset=['timestamp'])
            final_count = len(df_clean)
            actual_dropped = initial_count - final_count
        else:
            actual_dropped = 0
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            processor_type=self.data_type,
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=len(df_clean),
            dropped_records=actual_dropped,
            flagged_records=0,
            validation_rules=[
                RuleResult(
                    rule_name="timestamp_parsing",
                    rule_type=RuleType.SCHEMA,
                    passed=parsing_errors == 0,
                    affected_records=parsing_errors,
                    violation_details=[f"{parsing_errors} timestamps could not be parsed"] if parsing_errors > 0 else [],
                    action_taken="drop" if parsing_errors > 0 else "none"
                ),
                RuleResult(
                    rule_name="timestamp_normalization",
                    rule_type=RuleType.SCHEMA,
                    passed=True,
                    affected_records=normalized_count,
                    violation_details=[],
                    action_taken="transform"
                )
            ],
            metrics={
                'timestamp_normalization_duration': duration,
                'parsing_errors': parsing_errors,
                'normalized_count': normalized_count,
                'target_timezone': self.config.target_timezone,
                'hourly_rounding': self.data_type == 'weather'
            }
        )
        
        return df_clean, result
    
    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Remove duplicates for both train and weather data."""
        start_time = datetime.now()
        df_clean = df.copy()
        original_count = len(df_clean)
        
        if original_count == 0:
            return df_clean, ValidationResult(
                processor_type=self.data_type,
                timestamp=datetime.now(),
                total_records=0,
                valid_records=0,
                dropped_records=0,
                flagged_records=0
            )
        
        # Identify duplicates based on key fields
        key_fields = [field for field in self.dedup_key_fields if field in df_clean.columns]
        
        if not key_fields:
            logger.warning(f"No deduplication key fields found in {self.data_type} dataset")
            return df_clean, ValidationResult(
                processor_type=self.data_type,
                timestamp=datetime.now(),
                total_records=original_count,
                valid_records=original_count,
                dropped_records=0,
                flagged_records=0
            )
        
        # Different strategies for train vs weather
        if self.data_type == 'train':
            # For train data, keep record with highest completeness score
            def calculate_completeness(row):
                non_null_count = row.notna().sum()
                return non_null_count / len(row)
            
            df_clean['_completeness_score'] = df_clean.apply(calculate_completeness, axis=1)
            df_clean = df_clean.sort_values('_completeness_score', ascending=False)
            df_clean = df_clean.drop_duplicates(subset=key_fields, keep='first')
            df_clean = df_clean.drop(columns=['_completeness_score'])
            
        elif self.data_type == 'weather':
            # For weather data, keep last record (most recent)
            df_clean = df_clean.drop_duplicates(subset=key_fields, keep='last')
        
        final_count = len(df_clean)
        duplicate_count = original_count - final_count
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            processor_type=self.data_type,
            timestamp=datetime.now(),
            total_records=original_count,
            valid_records=final_count,
            dropped_records=duplicate_count,
            flagged_records=0,
            validation_rules=[
                RuleResult(
                    rule_name="duplicate_removal",
                    rule_type=RuleType.DEDUPLICATION,
                    passed=True,
                    affected_records=duplicate_count,
                    violation_details=[f"Removed {duplicate_count} duplicate {self.data_type} records"] if duplicate_count > 0 else [],
                    action_taken="drop" if duplicate_count > 0 else "none"
                )
            ],
            metrics={
                'deduplication_duration': duration,
                'duplicate_count': duplicate_count,
                'deduplication_rate': duplicate_count / original_count if original_count > 0 else 0.0,
                'key_fields': key_fields,
                'selection_strategy': 'completeness' if self.data_type == 'train' else 'keep_last'
            }
        )
        
        return df_clean, result
    
    def apply_business_rules(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
        """Apply business rules specific to data type."""
        start_time = datetime.now()
        df_clean = df.copy()
        total_records = len(df_clean)
        
        rules = []
        flagged_records = 0
        
        if self.data_type == 'train':
            # Train-specific business rules
            flagged_records = self._apply_train_business_rules(df_clean, rules)
        elif self.data_type == 'weather':
            # Weather-specific business rules
            flagged_records = self._apply_weather_business_rules(df_clean, rules)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = ValidationResult(
            processor_type=self.data_type,
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=len(df_clean),
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'business_rules_duration': duration,
                'data_type': self.data_type
            }
        )
        
        return df_clean, result
    
    def _apply_train_business_rules(self, df_clean: pd.DataFrame, rules: List[RuleResult]) -> int:
        """Apply train-specific business rules."""
        flagged_records = 0
        
        # Initialize flag columns
        if 'delay_minutes' in df_clean.columns:
            df_clean['delay_anomaly'] = 0
            df_clean['delay_outlier'] = 0
        
        if 'delay_minutes' in df_clean.columns:
            # Rule: Delays < -5 minutes are set to NULL and flagged
            extreme_early_mask = df_clean['delay_minutes'] < self.train_config.delay_min_threshold
            extreme_early_count = extreme_early_mask.sum()
            
            if extreme_early_count > 0:
                df_clean.loc[extreme_early_mask, 'delay_minutes'] = np.nan
                df_clean.loc[extreme_early_mask, 'delay_anomaly'] = 1
                flagged_records += extreme_early_count
                
                rules.append(RuleResult(
                    rule_name="extreme_early_delay",
                    rule_type=RuleType.BUSINESS,
                    passed=extreme_early_count == 0,
                    affected_records=extreme_early_count,
                    violation_details=[f"{extreme_early_count} records with delay < {self.train_config.delay_min_threshold} minutes"],
                    action_taken="transform"
                ))
            
            # Rule: Delays > 300 minutes are flagged as outliers but kept
            extreme_delay_mask = df_clean['delay_minutes'] > self.train_config.delay_max_threshold
            extreme_delay_count = extreme_delay_mask.sum()
            
            if extreme_delay_count > 0:
                df_clean.loc[extreme_delay_mask, 'delay_outlier'] = 1
                flagged_records += extreme_delay_count
                
                rules.append(RuleResult(
                    rule_name="extreme_delay_outlier",
                    rule_type=RuleType.BUSINESS,
                    passed=extreme_delay_count == 0,
                    affected_records=extreme_delay_count,
                    violation_details=[f"{extreme_delay_count} records with delay > {self.train_config.delay_max_threshold} minutes"],
                    action_taken="flag"
                ))
        
        return flagged_records
    
    def _apply_weather_business_rules(self, df_clean: pd.DataFrame, rules: List[RuleResult]) -> int:
        """Apply weather-specific business rules."""
        flagged_records = 0
        
        # Initialize flag columns
        if 'temperature' in df_clean.columns:
            df_clean['temp_range_violation'] = 0
        if 'precipitation' in df_clean.columns:
            df_clean['precip_range_violation'] = 0
        
        # Temperature range validation
        if 'temperature' in df_clean.columns:
            temp_min, temp_max = self.weather_config.temperature_range
            out_of_range_mask = (df_clean['temperature'] < temp_min) | (df_clean['temperature'] > temp_max)
            out_of_range_count = out_of_range_mask.sum()
            
            if out_of_range_count > 0:
                df_clean.loc[out_of_range_mask, 'temperature'] = np.nan
                df_clean.loc[out_of_range_mask, 'temp_range_violation'] = 1
                flagged_records += out_of_range_count
                
                rules.append(RuleResult(
                    rule_name="temperature_range_validation",
                    rule_type=RuleType.RANGE,
                    passed=out_of_range_count == 0,
                    affected_records=out_of_range_count,
                    violation_details=[f"{out_of_range_count} temperature values outside plausible range {temp_min}-{temp_max}°C"],
                    action_taken="transform"
                ))
        
        # Precipitation validation (must be >= 0)
        precip_fields = ['precipitation', 'precip_mm']
        for field in precip_fields:
            if field in df_clean.columns:
                negative_precip_mask = df_clean[field] < self.weather_config.precipitation_min
                negative_precip_count = negative_precip_mask.sum()
                
                if negative_precip_count > 0:
                    df_clean.loc[negative_precip_mask, field] = np.nan
                    df_clean.loc[negative_precip_mask, 'precip_range_violation'] = 1
                    flagged_records += negative_precip_count
                    
                    rules.append(RuleResult(
                        rule_name=f"{field}_range_validation",
                        rule_type=RuleType.RANGE,
                        passed=negative_precip_count == 0,
                        affected_records=negative_precip_count,
                        violation_details=[f"{negative_precip_count} negative {field} values (impossible)"],
                        action_taken="transform"
                    ))
        
        return flagged_records
    
    def process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[ValidationResult]]:
        """Execute complete data quality processing pipeline."""
        start_time = datetime.now()
        results = []
        
        logger.info(f"Starting {self.data_type} data quality processing for {len(df)} records")
        
        # 1. Schema validation
        schema_result = self.validate_schema(df)
        results.append(schema_result)
        
        if schema_result.dropped_records == schema_result.total_records:
            logger.error(f"All {self.data_type} records failed schema validation")
            return pd.DataFrame(), results
        
        # 2. Timestamp normalization
        df_clean, timestamp_result = self.normalize_timestamps(df)
        results.append(timestamp_result)
        
        # 3. Deduplication
        df_clean, dedup_result = self.remove_duplicates(df_clean)
        results.append(dedup_result)
        
        # 4. Business rules
        df_clean, business_result = self.apply_business_rules(df_clean)
        results.append(business_result)
        
        # Store results for summary
        self.validation_results.extend(results)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"{self.data_type.title()} data quality processing completed in {total_duration:.2f}s. "
                   f"Final dataset: {len(df_clean)} records")
        
        return df_clean, results


# Convenience functions for backward compatibility
def create_train_processor(config: Optional[TrainDataQualityConfig] = None) -> UnifiedDataQualityProcessor:
    """Create a unified processor configured for train data."""
    return UnifiedDataQualityProcessor('train', train_config=config)


def create_weather_processor(config: Optional[WeatherDataQualityConfig] = None) -> UnifiedDataQualityProcessor:
    """Create a unified processor configured for weather data."""
    return UnifiedDataQualityProcessor('weather', weather_config=config)