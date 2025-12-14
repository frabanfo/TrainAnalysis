"""
Post-Integration Data Quality Processor

This module validates the quality of integrated train-weather data after the integration process.
It checks for data completeness, consistency, and business rule compliance in the integrated dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger

from .base import ValidationResult, RuleResult, RuleType
from ..database.db_manager import DatabaseManager


class PostIntegrationQualityProcessor:
    """
    Data quality processor for integrated train-weather data.
    
    Validates the quality of data after train and weather data have been integrated,
    including feature engineering and storage in the database.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        
        # Expected columns in integrated data
        self.required_columns = [
            'train_id', 'timestamp', 'station_code', 'delay_minutes',
            'temperature', 'wind_speed', 'precip_mm', 'weather_code',
            'train_category', 'route', 'delay_status', 'destination', 'is_cancelled'
        ]
        
        # Feature engineering columns
        self.feature_columns = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'is_raining', 'rain_intensity', 'wind_category', 
            'is_delayed', 'delay_category'
        ]
        
        # Quality thresholds
        self.thresholds = {
            'min_weather_match_rate': 0.95,  # 95% of records should have weather data
            'max_null_rate': 0.05,  # Max 5% null values in critical fields
            'max_anomaly_rate': 0.02,  # Max 2% anomalous values
            'min_feature_completeness': 0.98  # 98% feature completeness
        }
    
    def assess_integration_quality(self, date: str) -> Dict[str, Any]:
        """
        Assess the quality of integrated data for a specific date.
        
        Args:
            date: Date to assess (YYYY-MM-DD format)
            
        Returns:
            Dictionary containing quality assessment results
        """
        logger.info(f"Starting post-integration quality assessment for {date}")
        start_time = datetime.now()
        
        try:
            # Load integrated data from database
            integrated_data = self._load_integrated_data(date)
            
            if integrated_data.empty:
                return {
                    'date': date,
                    'success': False,
                    'error': 'No integrated data found for the specified date',
                    'total_records': 0,
                    'assessment_results': []
                }
            
            # Run quality assessments
            results = []
            
            # 1. Data completeness assessment
            completeness_result = self._assess_data_completeness(integrated_data)
            results.append(completeness_result)
            
            # 2. Weather matching assessment
            weather_match_result = self._assess_weather_matching(integrated_data)
            results.append(weather_match_result)
            
            # 3. Feature engineering assessment
            feature_result = self._assess_feature_engineering(integrated_data)
            results.append(feature_result)
            
            # 4. Business logic consistency
            consistency_result = self._assess_business_consistency(integrated_data)
            results.append(consistency_result)
            
            # 5. Data anomaly detection
            anomaly_result = self._assess_data_anomalies(integrated_data)
            results.append(anomaly_result)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_score(results)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            return {
                'date': date,
                'success': True,
                'total_records': len(integrated_data),
                'overall_quality_score': overall_score,
                'assessment_duration': duration,
                'assessment_results': results,
                'summary': self._generate_summary(results, overall_score)
            }
            
        except Exception as e:
            logger.error(f"Post-integration quality assessment failed: {str(e)}")
            return {
                'date': date,
                'success': False,
                'error': str(e),
                'total_records': 0,
                'assessment_results': []
            }
    
    def _load_integrated_data(self, date: str) -> pd.DataFrame:
        """Load integrated data from database for the specified date."""
        try:
            # Use simple string formatting for PostgreSQL date comparison
            df = self.db_manager.execute_query(
                f"SELECT * FROM train_weather_integrated WHERE timestamp::date = '{date}' ORDER BY timestamp, station_code"
            )
            
            if df.empty:
                logger.warning(f"No integrated data found for date {date}")
            else:
                logger.info(f"Loaded {len(df)} integrated records for {date}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load integrated data for {date}: {str(e)}")
            return pd.DataFrame()
    
    def _assess_data_completeness(self, df: pd.DataFrame) -> ValidationResult:
        """Assess completeness of required fields in integrated data."""
        start_time = datetime.now()
        total_records = len(df)
        rules = []
        
        # Check presence of required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            rules.append(RuleResult(
                rule_name="required_columns_present",
                rule_type=RuleType.SCHEMA,
                passed=False,
                affected_records=total_records,
                violation_details=[f"Missing required columns: {missing_columns}"],
                action_taken="none"
            ))
        
        # Check null rates for critical fields
        critical_fields = ['train_id', 'timestamp', 'station_code']
        flagged_records = 0
        
        for field in critical_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                null_rate = null_count / total_records if total_records > 0 else 0
                
                rules.append(RuleResult(
                    rule_name=f"{field}_completeness",
                    rule_type=RuleType.COMPLETENESS,
                    passed=null_rate <= self.thresholds['max_null_rate'],
                    affected_records=null_count,
                    violation_details=[f"{field} has {null_rate:.2%} null values (threshold: {self.thresholds['max_null_rate']:.2%})"] if null_rate > self.thresholds['max_null_rate'] else [],
                    action_taken="flag" if null_rate > self.thresholds['max_null_rate'] else "none"
                ))
                
                if null_rate > self.thresholds['max_null_rate']:
                    flagged_records += null_count
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type="post_integration",
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=total_records - flagged_records,
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'completeness_duration': duration,
                'missing_columns': missing_columns,
                'critical_fields_checked': len(critical_fields)
            }
        )
    
    def _assess_weather_matching(self, df: pd.DataFrame) -> ValidationResult:
        """Assess quality of weather data matching."""
        start_time = datetime.now()
        total_records = len(df)
        rules = []
        
        # Check weather data availability
        weather_fields = ['temperature', 'wind_speed', 'precip_mm', 'weather_code']
        weather_available = df[weather_fields].notna().any(axis=1)
        weather_match_count = weather_available.sum()
        weather_match_rate = weather_match_count / total_records if total_records > 0 else 0
        
        rules.append(RuleResult(
            rule_name="weather_data_matching",
            rule_type=RuleType.CONSISTENCY,
            passed=weather_match_rate >= self.thresholds['min_weather_match_rate'],
            affected_records=total_records - weather_match_count,
            violation_details=[f"Weather match rate {weather_match_rate:.2%} below threshold {self.thresholds['min_weather_match_rate']:.2%}"] if weather_match_rate < self.thresholds['min_weather_match_rate'] else [],
            action_taken="flag" if weather_match_rate < self.thresholds['min_weather_match_rate'] else "none"
        ))
        
        # Check for reasonable weather values
        flagged_records = 0
        
        if 'temperature' in df.columns:
            temp_outliers = ((df['temperature'] < -50) | (df['temperature'] > 60)).sum()
            if temp_outliers > 0:
                flagged_records += temp_outliers
                rules.append(RuleResult(
                    rule_name="temperature_plausibility",
                    rule_type=RuleType.RANGE,
                    passed=temp_outliers == 0,
                    affected_records=temp_outliers,
                    violation_details=[f"{temp_outliers} temperature values outside plausible range (-50°C to 60°C)"],
                    action_taken="flag"
                ))
        
        if 'precip_mm' in df.columns:
            precip_outliers = (df['precip_mm'] < 0).sum()
            if precip_outliers > 0:
                flagged_records += precip_outliers
                rules.append(RuleResult(
                    rule_name="precipitation_validity",
                    rule_type=RuleType.RANGE,
                    passed=precip_outliers == 0,
                    affected_records=precip_outliers,
                    violation_details=[f"{precip_outliers} negative precipitation values (impossible)"],
                    action_taken="flag"
                ))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type="post_integration",
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=total_records - flagged_records,
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'weather_matching_duration': duration,
                'weather_match_rate': weather_match_rate,
                'weather_fields_checked': len(weather_fields)
            }
        )
    
    def _assess_feature_engineering(self, df: pd.DataFrame) -> ValidationResult:
        """Assess quality of feature engineering."""
        start_time = datetime.now()
        total_records = len(df)
        rules = []
        flagged_records = 0
        
        # Check presence of feature columns
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            rules.append(RuleResult(
                rule_name="feature_columns_present",
                rule_type=RuleType.SCHEMA,
                passed=False,
                affected_records=total_records,
                violation_details=[f"Missing feature columns: {missing_features}"],
                action_taken="flag"
            ))
            flagged_records = total_records
        
        # Check feature completeness
        available_features = [col for col in self.feature_columns if col in df.columns]
        if available_features:
            feature_completeness = df[available_features].notna().all(axis=1).sum() / total_records
            
            rules.append(RuleResult(
                rule_name="feature_completeness",
                rule_type=RuleType.COMPLETENESS,
                passed=feature_completeness >= self.thresholds['min_feature_completeness'],
                affected_records=int((1 - feature_completeness) * total_records),
                violation_details=[f"Feature completeness {feature_completeness:.2%} below threshold {self.thresholds['min_feature_completeness']:.2%}"] if feature_completeness < self.thresholds['min_feature_completeness'] else [],
                action_taken="flag" if feature_completeness < self.thresholds['min_feature_completeness'] else "none"
            ))
        
        # Validate specific features
        if 'hour_of_day' in df.columns:
            invalid_hours = ((df['hour_of_day'] < 0) | (df['hour_of_day'] > 23)).sum()
            if invalid_hours > 0:
                flagged_records += invalid_hours
                rules.append(RuleResult(
                    rule_name="hour_of_day_validity",
                    rule_type=RuleType.RANGE,
                    passed=invalid_hours == 0,
                    affected_records=invalid_hours,
                    violation_details=[f"{invalid_hours} invalid hour_of_day values (must be 0-23)"],
                    action_taken="flag"
                ))
        
        if 'day_of_week' in df.columns:
            invalid_days = ((df['day_of_week'] < 0) | (df['day_of_week'] > 6)).sum()
            if invalid_days > 0:
                flagged_records += invalid_days
                rules.append(RuleResult(
                    rule_name="day_of_week_validity",
                    rule_type=RuleType.RANGE,
                    passed=invalid_days == 0,
                    affected_records=invalid_days,
                    violation_details=[f"{invalid_days} invalid day_of_week values (must be 0-6)"],
                    action_taken="flag"
                ))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type="post_integration",
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=total_records - flagged_records,
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'feature_engineering_duration': duration,
                'available_features': len(available_features),
                'missing_features': len(missing_features)
            }
        )
    
    def _assess_business_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Assess business logic consistency in integrated data."""
        start_time = datetime.now()
        total_records = len(df)
        rules = []
        flagged_records = 0
        
        # Check delay consistency
        if 'delay_minutes' in df.columns and 'is_delayed' in df.columns:
            # Records with delay > 5 should be marked as delayed
            should_be_delayed = df['delay_minutes'] > 5
            marked_as_delayed = df['is_delayed'] == True
            
            inconsistent_delays = should_be_delayed & ~marked_as_delayed
            inconsistent_count = inconsistent_delays.sum()
            
            if inconsistent_count > 0:
                flagged_records += inconsistent_count
                rules.append(RuleResult(
                    rule_name="delay_flag_consistency",
                    rule_type=RuleType.BUSINESS,
                    passed=inconsistent_count == 0,
                    affected_records=inconsistent_count,
                    violation_details=[f"{inconsistent_count} records with delay > 5min not marked as delayed"],
                    action_taken="flag"
                ))
        
        # Check weather-rain consistency
        if 'precip_mm' in df.columns and 'is_raining' in df.columns:
            has_precipitation = df['precip_mm'] > 0
            marked_as_raining = df['is_raining'] == True
            
            inconsistent_rain = has_precipitation & ~marked_as_raining
            inconsistent_rain_count = inconsistent_rain.sum()
            
            if inconsistent_rain_count > 0:
                flagged_records += inconsistent_rain_count
                rules.append(RuleResult(
                    rule_name="rain_flag_consistency",
                    rule_type=RuleType.BUSINESS,
                    passed=inconsistent_rain_count == 0,
                    affected_records=inconsistent_rain_count,
                    violation_details=[f"{inconsistent_rain_count} records with precipitation > 0 not marked as raining"],
                    action_taken="flag"
                ))
        
        # Check weekend consistency
        if 'day_of_week' in df.columns and 'is_weekend' in df.columns:
            is_weekend_day = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
            marked_as_weekend = df['is_weekend'] == True
            
            inconsistent_weekend = is_weekend_day & ~marked_as_weekend
            inconsistent_weekend_count = inconsistent_weekend.sum()
            
            if inconsistent_weekend_count > 0:
                flagged_records += inconsistent_weekend_count
                rules.append(RuleResult(
                    rule_name="weekend_flag_consistency",
                    rule_type=RuleType.BUSINESS,
                    passed=inconsistent_weekend_count == 0,
                    affected_records=inconsistent_weekend_count,
                    violation_details=[f"{inconsistent_weekend_count} weekend days not marked as weekend"],
                    action_taken="flag"
                ))
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type="post_integration",
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=total_records - flagged_records,
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'consistency_duration': duration,
                'business_rules_checked': len(rules)
            }
        )
    
    def _assess_data_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """Detect anomalies in integrated data."""
        start_time = datetime.now()
        total_records = len(df)
        rules = []
        flagged_records = 0
        
        # Detect extreme delays
        if 'delay_minutes' in df.columns:
            extreme_delays = (df['delay_minutes'] > 300).sum()  # > 5 hours
            extreme_delay_rate = extreme_delays / total_records if total_records > 0 else 0
            
            rules.append(RuleResult(
                rule_name="extreme_delay_detection",
                rule_type=RuleType.BUSINESS,
                passed=extreme_delay_rate <= self.thresholds['max_anomaly_rate'],
                affected_records=extreme_delays,
                violation_details=[f"{extreme_delays} records with extreme delays (>300min), rate: {extreme_delay_rate:.2%}"] if extreme_delay_rate > self.thresholds['max_anomaly_rate'] else [],
                action_taken="flag" if extreme_delay_rate > self.thresholds['max_anomaly_rate'] else "none"
            ))
            
            if extreme_delay_rate > self.thresholds['max_anomaly_rate']:
                flagged_records += extreme_delays
        
        # Detect duplicate train records (same train_id, timestamp, station)
        if all(col in df.columns for col in ['train_id', 'timestamp', 'station_code']):
            duplicates = df.duplicated(subset=['train_id', 'timestamp', 'station_code']).sum()
            
            rules.append(RuleResult(
                rule_name="duplicate_detection",
                rule_type=RuleType.DEDUPLICATION,
                passed=duplicates == 0,
                affected_records=duplicates,
                violation_details=[f"{duplicates} duplicate records found"] if duplicates > 0 else [],
                action_taken="flag" if duplicates > 0 else "none"
            ))
            
            if duplicates > 0:
                flagged_records += duplicates
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return ValidationResult(
            processor_type="post_integration",
            timestamp=datetime.now(),
            total_records=total_records,
            valid_records=total_records - flagged_records,
            dropped_records=0,
            flagged_records=flagged_records,
            validation_rules=rules,
            metrics={
                'anomaly_detection_duration': duration,
                'anomaly_checks_performed': len(rules)
            }
        )
    
    def _calculate_overall_score(self, results: List[ValidationResult]) -> float:
        """Calculate overall quality score from validation results."""
        if not results:
            return 0.0
        
        total_records = results[0].total_records if results else 0
        if total_records == 0:
            return 0.0
        
        # Weight different aspects of quality
        weights = {
            'completeness': 0.3,
            'weather_matching': 0.25,
            'feature_engineering': 0.2,
            'consistency': 0.15,
            'anomaly_detection': 0.1
        }
        
        weighted_score = 0.0
        
        for i, result in enumerate(results):
            if result.total_records > 0:
                # Calculate success rate for this validation
                success_rate = result.valid_records / result.total_records
                
                # Apply weight based on validation type
                weight_key = list(weights.keys())[i] if i < len(weights) else 'anomaly_detection'
                weight = weights.get(weight_key, 0.1)
                
                weighted_score += success_rate * weight
        
        return min(weighted_score, 1.0)  # Cap at 1.0
    
    def _generate_summary(self, results: List[ValidationResult], overall_score: float) -> Dict[str, Any]:
        """Generate a summary of the quality assessment."""
        total_records = results[0].total_records if results else 0
        total_flagged = sum(result.flagged_records for result in results)
        
        # Categorize quality level
        if overall_score >= 0.95:
            quality_level = "EXCELLENT"
        elif overall_score >= 0.85:
            quality_level = "GOOD"
        elif overall_score >= 0.70:
            quality_level = "ACCEPTABLE"
        else:
            quality_level = "POOR"
        
        # Count issues by type
        issues_by_type = {}
        for result in results:
            for rule in result.validation_rules:
                if not rule.passed:
                    rule_type = rule.rule_type.value
                    if rule_type not in issues_by_type:
                        issues_by_type[rule_type] = 0
                    issues_by_type[rule_type] += rule.affected_records
        
        return {
            'quality_level': quality_level,
            'overall_score': overall_score,
            'total_records': total_records,
            'total_flagged_records': total_flagged,
            'flagged_rate': total_flagged / total_records if total_records > 0 else 0,
            'issues_by_type': issues_by_type,
            'recommendations': self._generate_recommendations(results, overall_score)
        }
    
    def _generate_recommendations(self, results: List[ValidationResult], overall_score: float) -> List[str]:
        """Generate recommendations based on quality assessment results."""
        recommendations = []
        
        if overall_score < 0.70:
            recommendations.append("CRITICAL: Overall data quality is poor. Review integration process.")
        
        for result in results:
            for rule in result.validation_rules:
                if not rule.passed and rule.affected_records > 0:
                    if rule.rule_name == "weather_data_matching":
                        recommendations.append("Improve weather data matching - check time synchronization and station coverage")
                    elif "consistency" in rule.rule_name:
                        recommendations.append("Review business logic implementation in feature engineering")
                    elif "completeness" in rule.rule_name:
                        recommendations.append("Address data completeness issues in source systems")
                    elif "anomaly" in rule.rule_name or "extreme" in rule.rule_name:
                        recommendations.append("Investigate and handle data anomalies and outliers")
        
        if not recommendations:
            recommendations.append("Data quality is good. Continue monitoring.")
        
        return recommendations