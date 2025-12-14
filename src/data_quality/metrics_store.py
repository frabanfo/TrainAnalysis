"""
Database implementation of quality metrics storage.

This module provides a concrete implementation of QualityMetricsStore
that uses the existing data_quality_metrics table for persistence.
"""

import json
import sys
import os
import numpy as np
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from loguru import logger

from .base import QualityMetricsStore, ValidationResult, QualityReport

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..database.db_manager import DatabaseManager
except ImportError:
    try:
        # Try absolute import
        from database.db_manager import DatabaseManager
    except ImportError:
        # Mock for testing when database is not available
        import pandas as pd
        
        class DatabaseManager:
            def execute_non_query(self, query, params): 
                return True
            def execute_query(self, query, params): 
                return pd.DataFrame()


class DatabaseQualityMetricsStore(QualityMetricsStore):
    """
    Unified database implementation for all quality metrics storage.
    
    Handles both individual ValidationResults and complete PreIntegrationQualityReports
    using the existing data_quality_metrics table.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the unified database metrics store.
        """
        self.db_manager = db_manager or DatabaseManager()
    
    def _serialize_for_json(self, obj):
        """
        Convert numpy types and other non-serializable objects to JSON-compatible types.
        """
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle custom objects by converting to dict
            return self._serialize_for_json(obj.__dict__)
        else:
            return obj
    
    def _safe_json_dumps(self, obj):
        """
        Safely serialize object to JSON, handling numpy types.
        """
        try:
            serializable_obj = self._serialize_for_json(obj)
            return json.dumps(serializable_obj)
        except Exception as e:
            logger.warning(f"JSON serialization failed, using fallback: {e}")
            # Fallback: convert to string representation
            return json.dumps(str(obj))
    
    def store_validation_result(self, result: ValidationResult) -> bool:
        """
        Store a validation result in the data_quality_metrics table.
        """
        try:
            # Store summary metrics
            summary_metrics = {
                'total_records': result.total_records,
                'valid_records': result.valid_records,
                'dropped_records': result.dropped_records,
                'flagged_records': result.flagged_records,
                'validation_success_rate': result.valid_records / result.total_records if result.total_records > 0 else 0.0
            }
            
            # Store each metric as a separate row
            for metric_name, metric_value in summary_metrics.items():
                details = {
                    'processor_type': result.processor_type,
                    'station_code': result.station_code,
                    'validation_rules': [
                        {
                            'rule_name': rule.rule_name,
                            'rule_type': rule.rule_type.value,
                            'passed': rule.passed,
                            'affected_records': rule.affected_records,
                            'action_taken': rule.action_taken
                        }
                        for rule in result.validation_rules
                    ],
                    'custom_metrics': result.metrics,
                    'anomalies': result.anomalies
                }
                
                query = """
                    INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, timestamp, details)
                    VALUES (:table_name, :metric_name, :metric_value, :timestamp, :details)
                """
                
                params = {
                    'table_name': result.processor_type,
                    'metric_name': metric_name,
                    'metric_value': float(metric_value),
                    'timestamp': result.timestamp,
                    'details': self._safe_json_dumps(details)
                }
                
                success = self.db_manager.execute_non_query(query, params)
                if not success:
                    logger.error(f"Failed to store metric {metric_name} for {result.processor_type}")
                    return False
            
            logger.info(f"Stored validation result for {result.processor_type} with {len(summary_metrics)} metrics")
            return True
            
        except Exception as e:
            logger.error(f"Error storing validation result: {str(e)}")
            return False
    
    def store_quality_report(self, report: QualityReport) -> bool:
        """
        Store a quality report in the data_quality_metrics table.
        """
        try:
            # Store report summary metrics
            for metric_name, metric_value in report.summary_metrics.items():
                details = {
                    'report_id': report.report_id,
                    'report_type': report.report_type,
                    'date_range': {
                        'start': report.date_range[0].isoformat(),
                        'end': report.date_range[1].isoformat()
                    },
                    'station_codes': report.station_codes,
                    'recommendations': report.recommendations,
                    'export_formats': report.export_formats,
                    'detailed_results_count': len(report.detailed_results)
                }
                
                query = """
                    INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, timestamp, details)
                    VALUES (:table_name, :metric_name, :metric_value, :timestamp, :details)
                """
                
                params = {
                    'table_name': f"{report.report_type}_report",
                    'metric_name': metric_name,
                    'metric_value': float(metric_value),
                    'timestamp': report.generation_timestamp,
                    'details': self._safe_json_dumps(details)
                }
                
                success = self.db_manager.execute_non_query(query, params)
                if not success:
                    logger.error(f"Failed to store report metric {metric_name} for {report.report_type}")
                    return False
            
            # Store a summary record for the entire report
            report_summary = {
                'total_validation_results': len(report.detailed_results),
                'report_completeness': 1.0 if report.detailed_results else 0.0
            }
            
            details = {
                'report_id': report.report_id,
                'report_type': report.report_type,
                'full_report': {
                    'date_range': {
                        'start': report.date_range[0].isoformat(),
                        'end': report.date_range[1].isoformat()
                    },
                    'station_codes': report.station_codes,
                    'summary_metrics': report.summary_metrics,
                    'recommendations': report.recommendations
                }
            }
            
            query = """
                INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, timestamp, details)
                VALUES (:table_name, :metric_name, :metric_value, :timestamp, :details)
            """
            
            params = {
                'table_name': f"{report.report_type}_report",
                'metric_name': 'report_summary',
                'metric_value': 1.0,
                'timestamp': report.generation_timestamp,
                'details': self._safe_json_dumps(details)
            }
            
            success = self.db_manager.execute_non_query(query, params)
            if not success:
                logger.error(f"Failed to store report summary for {report.report_type}")
                return False
            
            logger.info(f"Stored quality report {report.report_id} for {report.report_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing quality report: {str(e)}")
            return False
    
    def retrieve_reports(self, 
                        report_type: Optional[str] = None,
                        start_date: Optional[date] = None,
                        end_date: Optional[date] = None,
                        station_codes: Optional[List[str]] = None) -> List[QualityReport]:
        """
        Retrieve quality reports based on filters.
        """
        try:
            # Build query with filters
            where_conditions = ["metric_name = 'report_summary'"]
            params = {}
            
            if report_type:
                where_conditions.append("table_name = :table_name")
                params['table_name'] = f"{report_type}_report"
            
            if start_date:
                where_conditions.append("timestamp >= :start_date")
                params['start_date'] = datetime.combine(start_date, datetime.min.time())
            
            if end_date:
                where_conditions.append("timestamp <= :end_date")
                params['end_date'] = datetime.combine(end_date, datetime.max.time())
            
            query = f"""
                SELECT table_name, timestamp, details
                FROM data_quality_metrics
                WHERE {' AND '.join(where_conditions)}
                ORDER BY timestamp DESC
            """
            
            df = self.db_manager.execute_query(query, params)
            
            reports = []
            for _, row in df.iterrows():
                try:
                    details = json.loads(row['details'])
                    full_report = details.get('full_report', {})
                    
                    # Filter by station codes if specified
                    if station_codes:
                        report_stations = full_report.get('station_codes', [])
                        if not any(station in report_stations for station in station_codes):
                            continue
                    
                    # Reconstruct QualityReport object
                    date_range_data = full_report.get('date_range', {})
                    start_date_str = date_range_data.get('start')
                    end_date_str = date_range_data.get('end')
                    
                    if start_date_str and end_date_str:
                        date_range = (
                            datetime.fromisoformat(start_date_str).date(),
                            datetime.fromisoformat(end_date_str).date()
                        )
                    else:
                        date_range = (date.today(), date.today())
                    
                    report = QualityReport(
                        report_id=details.get('report_id', ''),
                        report_type=details.get('report_type', ''),
                        generation_timestamp=row['timestamp'],
                        date_range=date_range,
                        station_codes=full_report.get('station_codes', []),
                        summary_metrics=full_report.get('summary_metrics', {}),
                        detailed_results=[],  # Not stored in summary
                        recommendations=full_report.get('recommendations', [])
                    )
                    
                    reports.append(report)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse report details: {str(e)}")
                    continue
            
            logger.info(f"Retrieved {len(reports)} quality reports")
            return reports
            
        except Exception as e:
            logger.error(f"Error retrieving quality reports: {str(e)}")
            return []
    
    def get_latest_metrics(self, table_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get the latest metrics for a specific table/processor type.
        """
        try:
            query = """
                SELECT metric_name, metric_value, timestamp, details
                FROM data_quality_metrics
                WHERE table_name = :table_name
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            
            params = {'table_name': table_name, 'limit': limit}
            df = self.db_manager.execute_query(query, params)
            
            metrics = {}
            for _, row in df.iterrows():
                metrics[row['metric_name']] = {
                    'value': row['metric_value'],
                    'timestamp': row['timestamp'],
                    'details': json.loads(row['details']) if row['details'] else {}
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving latest metrics: {str(e)}")
            return {}
    
    # ========== Pre-Integration Quality Report Methods ==========
    
    def store_pre_integration_report(self, report) -> bool:
        """
        Store a pre-integration quality report in the database.
        
        Args:
            report: PreIntegrationQualityReport to store
            
        Returns:
            True if storage was successful, False otherwise
        """
        try:
            # Import here to avoid circular imports
            from .models import PreIntegrationQualityReport, QualityReportSerializer
            
            if not isinstance(report, PreIntegrationQualityReport):
                logger.error("Invalid report type for pre-integration storage")
                return False
            
            # Store main report summary
            summary_metrics = report.get_summary_metrics()
            
            # Store each summary metric as a separate row
            for metric_name, metric_value in summary_metrics.items():
                details = {
                    'report_id': report.report_id,
                    'processor_type': report.processor_type,
                    'date_range': {
                        'start': report.date_range[0].isoformat(),
                        'end': report.date_range[1].isoformat()
                    },
                    'station_codes': report.station_codes,
                    'validation_summary': {
                        'records_processed': report.validation_result.records_processed,
                        'records_valid': report.validation_result.records_valid,
                        'records_flagged': report.validation_result.records_flagged,
                        'records_dropped': report.validation_result.records_dropped
                    },
                    'four_pillars': {
                        'completeness_score': report.validation_result.completeness_score,
                        'accuracy_score': report.validation_result.accuracy_score,
                        'consistency_score': report.validation_result.consistency_score,
                        'uniqueness_score': report.validation_result.uniqueness_score,
                        'coverage_score': report.validation_result.coverage_score
                    },
                    'recommendations': report.recommendations,
                    'processing_duration': report.processing_duration
                }
                
                query = """
                    INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, timestamp, details)
                    VALUES (:table_name, :metric_name, :metric_value, :timestamp, :details)
                """
                
                params = {
                    'table_name': 'pre_integration',
                    'metric_name': metric_name,
                    'metric_value': float(metric_value),
                    'timestamp': report.generation_timestamp,
                    'details': self._safe_json_dumps(details)
                }
                
                success = self.db_manager.execute_non_query(query, params)
                if not success:
                    logger.error(f"Failed to store metric {metric_name} for pre-integration report")
                    return False
            
            # Store complete report as JSON
            report_json = QualityReportSerializer.to_json(report)
            complete_details = {
                'report_type': 'complete_pre_integration_report',
                'report_data': json.loads(report_json)
            }
            
            query = """
                INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, timestamp, details)
                VALUES (:table_name, :metric_name, :metric_value, :timestamp, :details)
            """
            
            params = {
                'table_name': 'pre_integration_complete',
                'metric_name': 'complete_report',
                'metric_value': report.validation_result.overall_quality_score,
                'timestamp': report.generation_timestamp,
                'details': self._safe_json_dumps(complete_details)
            }
            
            success = self.db_manager.execute_non_query(query, params)
            if not success:
                logger.error("Failed to store complete pre-integration report")
                return False
            
            logger.info(f"Stored pre-integration quality report {report.report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing pre-integration quality report: {str(e)}")
            return False
    
    def retrieve_pre_integration_reports(self, 
                                       start_date: Optional[date] = None,
                                       end_date: Optional[date] = None,
                                       station_codes: Optional[List[str]] = None,
                                       limit: int = 10):
        """
        Retrieve pre-integration quality reports based on filters.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            station_codes: Optional list of station codes to filter by
            limit: Maximum number of reports to retrieve
            
        Returns:
            List of PreIntegrationQualityReport objects
        """
        try:
            # Import here to avoid circular imports
            from .models import QualityReportSerializer
            from datetime import datetime
            
            # Build query with filters
            where_conditions = ["table_name = 'pre_integration_complete'", "metric_name = 'complete_report'"]
            params = {'limit': limit}
            
            if start_date:
                where_conditions.append("timestamp >= :start_date")
                params['start_date'] = datetime.combine(start_date, datetime.min.time())
            
            if end_date:
                where_conditions.append("timestamp <= :end_date")
                params['end_date'] = datetime.combine(end_date, datetime.max.time())
            
            query = f"""
                SELECT timestamp, details
                FROM data_quality_metrics
                WHERE {' AND '.join(where_conditions)}
                ORDER BY timestamp DESC
                LIMIT :limit
            """
            
            df = self.db_manager.execute_query(query, params)
            
            reports = []
            for _, row in df.iterrows():
                try:
                    details = json.loads(row['details'])
                    report_data = details.get('report_data', {})
                    
                    # Filter by station codes if specified
                    if station_codes:
                        report_stations = report_data.get('station_codes', [])
                        if not any(station in report_stations for station in station_codes):
                            continue
                    
                    # Reconstruct PreIntegrationQualityReport object
                    report = QualityReportSerializer.from_dict(report_data)
                    reports.append(report)
                    
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse report details: {str(e)}")
                    continue
            
            logger.info(f"Retrieved {len(reports)} pre-integration quality reports")
            return reports
            
        except Exception as e:
            logger.error(f"Error retrieving pre-integration quality reports: {str(e)}")
            return []