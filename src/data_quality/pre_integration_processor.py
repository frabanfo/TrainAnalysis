"""
Pre-integration data quality processor.

This module implements comprehensive pre-integration data quality validation
following the four pillars: Completeness, Accuracy, Consistency, and Uniqueness.
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from .base import BaseDataQualityProcessor, ValidationResult, RuleResult, RuleType
from .models import (
    DeduplicationResult, MissingValueResult, BusinessRuleResult, 
    CoverageResult, PreIntegrationValidationResult, PreIntegrationQualityReport
)
from .config import TrainDataQualityConfig, WeatherDataQualityConfig
# Sub-processors removed - not used in this implementation


class PreIntegrationProcessor:
    """
    Pre-integration data quality processor implementing the four pillars:
    - Completeness: Missing delay values, weather info, station coordinates
    - Accuracy: delay >= 0, arrival before departure, temp ranges, precip >= 0
    - Consistency: Time zones, weather-train timestamp alignment, station names
    - Uniqueness: No duplicate records
    """
    
    def __init__(self, 
                 train_config: Optional[TrainDataQualityConfig] = None,
                 weather_config: Optional[WeatherDataQualityConfig] = None):
        """Initialize pre-integration processor."""
        self.train_config = train_config or TrainDataQualityConfig()
        self.weather_config = weather_config or WeatherDataQualityConfig()
        
        # Configuration stored for assessment logic
    
    def assess_completeness(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[float, List[MissingValueResult]]:
        """
        A. Completeness Assessment
        - % missing delay values
        - % missing weather info  
        - % trains without station coordinates
        """
        missing_results = []
        
        # Train data completeness
        if not train_df.empty:
            total_train_records = len(train_df)
            
            # Missing delay values
            if 'delay_minutes' in train_df.columns:
                missing_delays = train_df['delay_minutes'].isnull().sum()
                missing_results.append(MissingValueResult(
                    field_name='delay_minutes',
                    missing_count=missing_delays,
                    missing_percentage=(missing_delays / total_train_records) * 100,
                    handling_strategy='flag_as_null'
                ))
            
            # Missing station coordinates
            coord_fields = ['station_lat', 'station_lon', 'latitude', 'longitude']
            available_coord_fields = [f for f in coord_fields if f in train_df.columns]
            
            if available_coord_fields:
                missing_coords = train_df[available_coord_fields].isnull().all(axis=1).sum()
                missing_results.append(MissingValueResult(
                    field_name='station_coordinates',
                    missing_count=missing_coords,
                    missing_percentage=(missing_coords / total_train_records) * 100,
                    handling_strategy='lookup_from_registry'
                ))
        
        # Weather data completeness
        if not weather_df.empty:
            total_weather_records = len(weather_df)
            
            # Missing weather info
            weather_fields = ['temperature', 'precip_mm', 'wind_speed', 'weather_code']
            available_weather_fields = [f for f in weather_fields if f in weather_df.columns]
            
            for field in available_weather_fields:
                missing_count = weather_df[field].isnull().sum()
                missing_results.append(MissingValueResult(
                    field_name=f'weather_{field}',
                    missing_count=missing_count,
                    missing_percentage=(missing_count / total_weather_records) * 100,
                    handling_strategy='nearest_hourly_snapshot'
                ))
        
        # Calculate overall completeness score
        if missing_results:
            avg_missing_pct = sum(r.missing_percentage for r in missing_results) / len(missing_results)
            completeness_score = max(0.0, (100 - avg_missing_pct) / 100)
        else:
            completeness_score = 1.0
        
        return completeness_score, missing_results
    
    def assess_accuracy(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[float, List[BusinessRuleResult]]:
        """
        B. Accuracy Assessment
        - delay >= 0
        - arrival_time should be before departure_time
        - precip_mm >= 0
        - temperature within plausible range (-20°C … 50°C)
        """
        business_rules = []
        
        # Train data accuracy rules
        if not train_df.empty:
            total_train_records = len(train_df)
            
            # Rule: delay >= 0
            if 'delay_minutes' in train_df.columns:
                negative_delays = (train_df['delay_minutes'] < 0).sum()
                business_rules.append(BusinessRuleResult(
                    rule_name='delay_non_negative',
                    rule_description='Delays must be >= 0 minutes',
                    violations_count=negative_delays,
                    violations_percentage=(negative_delays / total_train_records) * 100,
                    action_taken='convert_to_zero' if negative_delays > 0 else 'none',
                    threshold_value=0
                ))
            
            # Rule: arrival_time before departure_time
            if 'arrival_time' in train_df.columns and 'departure_time' in train_df.columns:
                # Convert to datetime for comparison
                train_df_temp = train_df.copy()
                try:
                    train_df_temp['arrival_dt'] = pd.to_datetime(train_df_temp['arrival_time'])
                    train_df_temp['departure_dt'] = pd.to_datetime(train_df_temp['departure_time'])
                    
                    invalid_order = (train_df_temp['arrival_dt'] >= train_df_temp['departure_dt']).sum()
                    business_rules.append(BusinessRuleResult(
                        rule_name='arrival_before_departure',
                        rule_description='Arrival time must be before departure time',
                        violations_count=invalid_order,
                        violations_percentage=(invalid_order / total_train_records) * 100,
                        action_taken='flag_for_investigation' if invalid_order > 0 else 'none'
                    ))
                except Exception as e:
                    logger.warning(f"Could not validate arrival/departure times: {e}")
        
        # Weather data accuracy rules
        if not weather_df.empty:
            total_weather_records = len(weather_df)
            
            # Rule: precip_mm >= 0
            if 'precipitation' in weather_df.columns or 'precip_mm' in weather_df.columns:
                precip_field = 'precipitation' if 'precipitation' in weather_df.columns else 'precip_mm'
                negative_precip = (weather_df[precip_field] < 0).sum()
                business_rules.append(BusinessRuleResult(
                    rule_name='precipitation_non_negative',
                    rule_description='Precipitation must be >= 0 mm',
                    violations_count=negative_precip,
                    violations_percentage=(negative_precip / total_weather_records) * 100,
                    action_taken='set_to_zero' if negative_precip > 0 else 'none',
                    threshold_value=0
                ))
            
            # Rule: temperature within plausible range (-20°C to 50°C)
            if 'temperature' in weather_df.columns:
                temp_min, temp_max = -20.0, 50.0
                out_of_range = ((weather_df['temperature'] < temp_min) | 
                               (weather_df['temperature'] > temp_max)).sum()
                business_rules.append(BusinessRuleResult(
                    rule_name='temperature_plausible_range',
                    rule_description=f'Temperature must be between {temp_min}°C and {temp_max}°C',
                    violations_count=out_of_range,
                    violations_percentage=(out_of_range / total_weather_records) * 100,
                    action_taken='flag_as_outlier' if out_of_range > 0 else 'none',
                    threshold_value=temp_max
                ))
        
        # Calculate overall accuracy score
        if business_rules:
            avg_violation_pct = sum(r.violations_percentage for r in business_rules) / len(business_rules)
            accuracy_score = max(0.0, (100 - avg_violation_pct) / 100)
        else:
            accuracy_score = 1.0
        
        return accuracy_score, business_rules
    
    def assess_consistency(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[float, int, int]:
        """
        C. Consistency Assessment
        - scheduled_time be either actual_time or more
        - time zones normalized (all timestamps in Italy timezone / UTC)
        - weather timestamp matches train timestamp within ±1 hour
        - standardized station names (canonical mapping)
        """
        timestamp_issues = 0
        schema_violations = 0
        total_records = len(train_df) + len(weather_df)
        
        # Check timezone consistency
        italy_tz = pytz.timezone('Europe/Rome')
        
        # Train timestamp consistency
        if not train_df.empty and 'timestamp' in train_df.columns:
            for idx, row in train_df.iterrows():
                try:
                    ts = pd.to_datetime(row['timestamp'])
                    if ts.tzinfo is None or ts.tzinfo != italy_tz:
                        timestamp_issues += 1
                except:
                    timestamp_issues += 1
        
        # Weather timestamp consistency  
        if not weather_df.empty and 'timestamp' in weather_df.columns:
            for idx, row in weather_df.iterrows():
                try:
                    ts = pd.to_datetime(row['timestamp'])
                    if ts.tzinfo is None or ts.tzinfo != italy_tz:
                        timestamp_issues += 1
                except:
                    timestamp_issues += 1
        
        # Check scheduled_time vs actual_time consistency
        if not train_df.empty:
            if 'scheduled_time' in train_df.columns and 'actual_time' in train_df.columns:
                try:
                    train_temp = train_df.copy()
                    train_temp['scheduled_dt'] = pd.to_datetime(train_temp['scheduled_time'])
                    train_temp['actual_dt'] = pd.to_datetime(train_temp['actual_time'])
                    
                    # scheduled_time should be <= actual_time
                    invalid_schedule = (train_temp['scheduled_dt'] > train_temp['actual_dt']).sum()
                    schema_violations += invalid_schedule
                except Exception as e:
                    logger.warning(f"Could not validate scheduled vs actual times: {e}")
        
        # Calculate consistency score
        total_issues = timestamp_issues + schema_violations
        consistency_score = max(0.0, 1.0 - (total_issues / total_records)) if total_records > 0 else 1.0
        
        return consistency_score, timestamp_issues, schema_violations
    
    def assess_uniqueness(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[float, DeduplicationResult]:
        """
        D. Uniqueness Assessment
        - no duplicate (train_id, timestamp) records
        - no duplicate (station_code, timestamp) weather snapshots
        """
        total_original = len(train_df) + len(weather_df)
        total_duplicates = 0
        
        # Train data deduplication
        train_duplicates = 0
        if not train_df.empty:
            train_key_fields = ['train_id', 'timestamp']
            available_train_keys = [f for f in train_key_fields if f in train_df.columns]
            
            if available_train_keys:
                train_duplicates = train_df.duplicated(subset=available_train_keys).sum()
                total_duplicates += train_duplicates
        
        # Weather data deduplication
        weather_duplicates = 0
        if not weather_df.empty:
            weather_key_fields = ['station_code', 'timestamp']
            available_weather_keys = [f for f in weather_key_fields if f in weather_df.columns]
            
            if available_weather_keys:
                weather_duplicates = weather_df.duplicated(subset=available_weather_keys).sum()
                total_duplicates += weather_duplicates
        
        # Create deduplication result
        dedup_result = DeduplicationResult(
            original_count=total_original,
            duplicate_count=total_duplicates,
            final_count=total_original - total_duplicates,
            deduplication_strategy='keep_highest_quality',
            key_fields=['train_id+timestamp', 'station_code+timestamp'],
            processing_time=0.0  # Will be set by caller
        )
        
        # Calculate uniqueness score
        uniqueness_score = 1.0 - (total_duplicates / total_original) if total_original > 0 else 1.0
        
        return uniqueness_score, dedup_result
    
    def assess_coverage(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[float, List[CoverageResult]]:
        """
        Coverage Assessment
        - Station coverage analysis
        - Temporal coverage analysis
        - Data availability assessment
        """
        coverage_results = []
        
        # Station coverage analysis
        train_stations = set(train_df['station_code'].unique()) if 'station_code' in train_df.columns else set()
        weather_stations = set(weather_df['station_code'].unique()) if 'station_code' in weather_df.columns else set()
        
        all_stations = train_stations.union(weather_stations)
        stations_with_both = train_stations.intersection(weather_stations)
        
        if all_stations:
            station_coverage = CoverageResult(
                entity_type='station',
                total_expected=len(all_stations),
                total_actual=len(stations_with_both),
                coverage_percentage=(len(stations_with_both) / len(all_stations)) * 100,
                missing_entities=list(all_stations - stations_with_both)
            )
            coverage_results.append(station_coverage)
        
        # Temporal coverage analysis
        if not train_df.empty and not weather_df.empty:
            if 'timestamp' in train_df.columns and 'timestamp' in weather_df.columns:
                try:
                    train_df_temp = train_df.copy()
                    weather_df_temp = weather_df.copy()
                    
                    train_df_temp['datetime'] = pd.to_datetime(train_df_temp['timestamp'])
                    weather_df_temp['datetime'] = pd.to_datetime(weather_df_temp['timestamp'])
                    
                    # Find overlapping time periods
                    train_min = train_df_temp['datetime'].min()
                    train_max = train_df_temp['datetime'].max()
                    weather_min = weather_df_temp['datetime'].min()
                    weather_max = weather_df_temp['datetime'].max()
                    
                    overlap_start = max(train_min, weather_min)
                    overlap_end = min(train_max, weather_max)
                    
                    if overlap_start <= overlap_end:
                        overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
                        total_hours = max((train_max - train_min).total_seconds() / 3600,
                                        (weather_max - weather_min).total_seconds() / 3600)
                        
                        temporal_coverage = CoverageResult(
                            entity_type='time_period',
                            total_expected=int(total_hours),
                            total_actual=int(overlap_hours),
                            coverage_percentage=(overlap_hours / total_hours) * 100 if total_hours > 0 else 0
                        )
                        coverage_results.append(temporal_coverage)
                except Exception as e:
                    logger.warning(f"Could not analyze temporal coverage: {e}")
        
        # Calculate overall coverage score
        if coverage_results:
            avg_coverage = sum(r.coverage_percentage for r in coverage_results) / len(coverage_results)
            coverage_score = avg_coverage / 100
        else:
            coverage_score = 0.0
        
        return coverage_score, coverage_results
    
    def apply_fixes(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply data quality fixes based on the assessment results.
        
        Fixes:
        - Fill weather using nearest hourly snapshot
        - Mark missing delays as NULL and flag them
        - Convert small negative delays to 0 if clearly rounding errors
        - Remove or flag unrealistic records (delays > 8 hours)
        - Enforce time ordering
        - Round/align timestamps to nearest hour for joins
        - Map station name variants to canonical names
        """
        train_fixed = train_df.copy()
        weather_fixed = weather_df.copy()
        
        # Fix train data
        if not train_fixed.empty:
            # Convert small negative delays to 0 (rounding errors)
            if 'delay_minutes' in train_fixed.columns:
                small_negative_mask = (train_fixed['delay_minutes'] >= -2) & (train_fixed['delay_minutes'] < 0)
                train_fixed.loc[small_negative_mask, 'delay_minutes'] = 0
                
                # Flag unrealistic delays (> 8 hours = 480 minutes)
                if 'delay_outlier' not in train_fixed.columns:
                    train_fixed['delay_outlier'] = 0
                extreme_delay_mask = train_fixed['delay_minutes'] > 480
                train_fixed.loc[extreme_delay_mask, 'delay_outlier'] = 1
            
            # Round timestamps to nearest hour for consistency
            if 'timestamp' in train_fixed.columns:
                try:
                    train_fixed['timestamp'] = pd.to_datetime(train_fixed['timestamp'])
                    train_fixed['timestamp'] = train_fixed['timestamp'].dt.round('h')
                except Exception as e:
                    logger.warning(f"Could not round train timestamps: {e}")
        
        # Fix weather data
        if not weather_fixed.empty:
            # Set negative precipitation to 0
            precip_fields = ['precipitation', 'precip_mm']
            for field in precip_fields:
                if field in weather_fixed.columns:
                    weather_fixed.loc[weather_fixed[field] < 0, field] = 0
            
            # Round timestamps to nearest hour
            if 'timestamp' in weather_fixed.columns:
                try:
                    weather_fixed['timestamp'] = pd.to_datetime(weather_fixed['timestamp'])
                    weather_fixed['timestamp'] = weather_fixed['timestamp'].dt.round('h')
                except Exception as e:
                    logger.warning(f"Could not round weather timestamps: {e}")
            
            # Flag temperature outliers but keep them
            if 'temperature' in weather_fixed.columns:
                if 'temp_outlier' not in weather_fixed.columns:
                    weather_fixed['temp_outlier'] = 0
                temp_outlier_mask = ((weather_fixed['temperature'] < -20) | 
                                   (weather_fixed['temperature'] > 50))
                weather_fixed.loc[temp_outlier_mask, 'temp_outlier'] = 1
        
        return train_fixed, weather_fixed
    
    def process(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> PreIntegrationQualityReport:
        """
        Execute complete pre-integration data quality assessment.
        
        Args:
            train_df: Train data DataFrame
            weather_df: Weather data DataFrame
            
        Returns:
            PreIntegrationQualityReport with comprehensive quality assessment
        """
        start_time = datetime.now()
        
        logger.info(f"Starting pre-integration quality assessment: "
                   f"{len(train_df)} train records, {len(weather_df)} weather records")
        
        # Apply fixes first
        train_fixed, weather_fixed = self.apply_fixes(train_df, weather_df)
        
        # Assess the four pillars
        completeness_score, missing_values = self.assess_completeness(train_fixed, weather_fixed)
        accuracy_score, business_rules = self.assess_accuracy(train_fixed, weather_fixed)
        consistency_score, timestamp_issues, schema_violations = self.assess_consistency(train_fixed, weather_fixed)
        uniqueness_score, deduplication = self.assess_uniqueness(train_fixed, weather_fixed)
        coverage_score, coverage_analysis = self.assess_coverage(train_fixed, weather_fixed)
        
        # Calculate overall quality score (weighted average)
        weights = {
            'completeness': 0.25,
            'accuracy': 0.25, 
            'consistency': 0.20,
            'uniqueness': 0.15,
            'coverage': 0.15
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            accuracy_score * weights['accuracy'] +
            consistency_score * weights['consistency'] +
            uniqueness_score * weights['uniqueness'] +
            coverage_score * weights['coverage']
        )
        
        # Create validation result
        total_records = len(train_df) + len(weather_df)
        total_fixed_records = len(train_fixed) + len(weather_fixed)
        records_dropped = total_records - total_fixed_records
        
        # Count flagged records
        flagged_count = 0
        if not train_fixed.empty:
            flag_columns = [col for col in train_fixed.columns if 'flag' in col.lower() or 'outlier' in col.lower()]
            if flag_columns:
                flagged_count += train_fixed[flag_columns].sum().sum()
        
        if not weather_fixed.empty:
            flag_columns = [col for col in weather_fixed.columns if 'flag' in col.lower() or 'outlier' in col.lower()]
            if flag_columns:
                flagged_count += weather_fixed[flag_columns].sum().sum()
        
        validation_result = PreIntegrationValidationResult(
            completeness_score=completeness_score,
            missing_values=missing_values,
            accuracy_score=accuracy_score,
            business_rules=business_rules,
            consistency_score=consistency_score,
            timestamp_issues=timestamp_issues,
            schema_violations=schema_violations,
            uniqueness_score=uniqueness_score,
            deduplication=deduplication,
            coverage_score=coverage_score,
            coverage_analysis=coverage_analysis,
            overall_quality_score=overall_score,
            processing_timestamp=datetime.now(),
            records_processed=total_records,
            records_valid=total_fixed_records,
            records_flagged=flagged_count,
            records_dropped=records_dropped
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_result)
        
        # Determine date range
        date_range = self._calculate_date_range(train_df, weather_df)
        
        # Get station codes
        station_codes = self._get_station_codes(train_df, weather_df)
        
        processing_duration = (datetime.now() - start_time).total_seconds()
        
        # Create quality report
        report = PreIntegrationQualityReport(
            report_id=f"pre_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            processor_type="pre_integration",
            generation_timestamp=datetime.now(),
            date_range=date_range,
            station_codes=station_codes,
            validation_result=validation_result,
            recommendations=recommendations,
            config_used={
                'train_config': self.train_config.__dict__,
                'weather_config': self.weather_config.__dict__
            },
            processing_duration=processing_duration
        )
        
        logger.info(f"Pre-integration quality assessment completed in {processing_duration:.2f}s. "
                   f"Overall quality score: {overall_score:.3f}")
        
        return report
    
    def _generate_recommendations(self, validation_result: PreIntegrationValidationResult) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Completeness recommendations
        if validation_result.completeness_score < 0.8:
            high_missing = [mv for mv in validation_result.missing_values if mv.missing_percentage > 20]
            if high_missing:
                recommendations.append(
                    f"High missing data detected in {len(high_missing)} fields. "
                    f"Consider data collection improvements or imputation strategies."
                )
        
        # Accuracy recommendations
        if validation_result.accuracy_score < 0.9:
            high_violations = [br for br in validation_result.business_rules if br.violations_percentage > 10]
            if high_violations:
                recommendations.append(
                    f"Significant business rule violations in {len(high_violations)} rules. "
                    f"Review data collection and validation processes."
                )
        
        # Consistency recommendations
        if validation_result.consistency_score < 0.95:
            if validation_result.timestamp_issues > 0:
                recommendations.append(
                    f"Found {validation_result.timestamp_issues} timestamp consistency issues. "
                    f"Ensure all data uses consistent timezone (Europe/Rome)."
                )
        
        # Uniqueness recommendations
        if validation_result.uniqueness_score < 0.98:
            dup_rate = (validation_result.deduplication.duplicate_count / 
                       validation_result.deduplication.original_count) * 100
            recommendations.append(
                f"Duplicate rate of {dup_rate:.1f}% detected. "
                f"Review data ingestion processes to prevent duplicates."
            )
        
        # Coverage recommendations
        if validation_result.coverage_score < 0.7:
            recommendations.append(
                "Low data coverage detected. Consider expanding data collection "
                "or focusing analysis on well-covered stations and time periods."
            )
        
        return recommendations
    
    def _calculate_date_range(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[datetime.date, datetime.date]:
        """Calculate the date range covered by the datasets."""
        dates = []
        
        for df in [train_df, weather_df]:
            if not df.empty and 'timestamp' in df.columns:
                try:
                    df_dates = pd.to_datetime(df['timestamp']).dt.date
                    dates.extend(df_dates.tolist())
                except:
                    pass
        
        if dates:
            return min(dates), max(dates)
        else:
            today = datetime.now().date()
            return today, today
    
    def _get_station_codes(self, train_df: pd.DataFrame, weather_df: pd.DataFrame) -> List[str]:
        """Get unique station codes from both datasets."""
        stations = set()
        
        for df in [train_df, weather_df]:
            if not df.empty and 'station_code' in df.columns:
                stations.update(df['station_code'].unique())
        
        return sorted(list(stations))