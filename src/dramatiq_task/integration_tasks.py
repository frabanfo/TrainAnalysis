import dramatiq
from datetime import datetime, timedelta
from typing import Dict, Any, List
from loguru import logger

from .dramatiq_config import PROCESSING_QUEUE
from src.data_integration import TrainWeatherIntegrator
from .logging_config import setup_task_logging


@dramatiq.actor(queue_name=PROCESSING_QUEUE, max_retries=3, min_backoff=30000, max_backoff=300000, store_results=True)
def integrate_train_weather_data(start_date: str, end_date: str, integration_id: str = None) -> Dict[str, Any]:
    log_file = setup_task_logging("integration", integration_id)
    
    try:
        logger.info(f"Starting train-weather integration task: {start_date} to {end_date} (integration_id: {integration_id})")
        logger.info(f"Task logs saved to: {log_file}")
        
        integrator = TrainWeatherIntegrator()
        
        if not integration_id:
            integration_id = f"integration_{start_date}_{end_date}_{int(datetime.now().timestamp())}"
        
        integration_result = integrator.integrate_data(start_date, end_date, integration_id)
        
        if integration_result['success']:
            task_result = {
                'task_type': 'train_weather_integration',
                'integration_id': integration_id,
                'start_date': start_date,
                'end_date': end_date,
                'total_records': int(integration_result['integrated_data']['total_records']),
                'source_records': {
                    'train_count': int(integration_result['source_data']['train_records']),
                    'weather_count': int(integration_result['source_data']['weather_records'])
                },
                'match_rate': float(integration_result['integrated_data']['match_rate']),
                'quality_score': float(integration_result['integrated_data']['quality_score']),
                'duration_seconds': float(integration_result['performance']['duration_seconds']),
                'records_per_second': float(integration_result['performance']['records_per_second']),
                'storage_success': integration_result['storage_success'],
                'failed_dates': [],
                'success': True,
                'skipped': integration_result.get('skipped', False),
                'skip_reason': integration_result.get('reason', None)
            }
        else:
            task_result = {
                'task_type': 'train_weather_integration',
                'integration_id': integration_id,
                'start_date': start_date,
                'end_date': end_date,
                'total_records': 0,
                'source_records': {'train_count': 0, 'weather_count': 0},
                'match_rate': 0.0,
                'quality_score': 0.0,
                'duration_seconds': float(integration_result.get('performance', {}).get('duration_seconds', 0)),
                'records_per_second': 0.0,
                'storage_success': False,
                'failed_dates': [start_date],
                'success': False,
                'error': integration_result.get('error', 'Integration failed')
            }
        
        logger.info(f"Integration task completed: {task_result}")
        return task_result
        
    except Exception as e:
        logger.error(f"Integration task failed: {str(e)}")
        
        failed_result = {
            'task_type': 'train_weather_integration',
            'integration_id': integration_id or f"failed_{int(datetime.now().timestamp())}",
            'start_date': start_date,
            'end_date': end_date,
            'total_records': 0,
            'source_records': {'train_count': 0, 'weather_count': 0},
            'match_rate': 0.0,
            'quality_score': 0.0,
            'duration_seconds': 0.0,
            'records_per_second': 0.0,
            'storage_success': False,
            'failed_dates': [start_date],
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        
        return failed_result
