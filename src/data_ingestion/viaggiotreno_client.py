"""
Viaggiotreno API client for collecting train delay data
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from data_ingestion.fetch_lombardia_stations import LombardiaStationsFetcher;

class ViaggiotrenoClient:
    """Client for Viaggiotreno API to collect train data with rate limiting"""
    
    def __init__(self):
        self.base_url = "http://www.viaggiatreno.it/infomobilita/resteasy/viaggiatreno"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting - Conservative approach for unofficial API
        self.min_request_interval = 2.0  # 2 seconds between requests
        self.requests_per_hour = 1800  # 30 requests per minute max
        self.requests_this_hour = 0
        self.current_hour = datetime.now().hour
        self.last_request_time = 0
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Major Italian stations - convert DataFrame to list of dicts with proper column mapping
        stations_df = LombardiaStationsFetcher().get_stored_stations()
        if not stations_df.empty:
            # Rename columns to match expected format
            stations_df = stations_df.rename(columns={
                'station_code': 'code',
                'station_name': 'name', 
                'latitude': 'lat',
                'longitude': 'lon'
            })
            self.stations = stations_df.to_dict('records')
        else:
            self.stations = []
    
    def collect_data(self, start_date: datetime, end_date: datetime, auto_resume: bool = True) -> List[Dict[Any, Any]]:
        """
        Collect train data for the specified date range with robust rate limiting and auto-resume
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            auto_resume: If True, wait for hourly limit reset and continue
            
        Returns:
            List of train records
        """
        all_data = []
        current_date = start_date
        total_days = (end_date - start_date).days + 1
        
        logger.info(f"Starting collection for {total_days} days from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        if auto_resume:
            logger.info("Auto-resume enabled: will wait for hourly limit reset if needed")
        
        while current_date <= end_date:
            # Check if we should continue
            if self.consecutive_errors >= self.max_consecutive_errors:
                if auto_resume:
                    logger.warning(f"Too many consecutive errors ({self.consecutive_errors}). Waiting 1 hour before retry...")
                    time.sleep(3600)
                    self.consecutive_errors = 0  # Reset after waiting
                    continue
                else:
                    logger.error(f"Too many consecutive errors ({self.consecutive_errors}). Stopping collection.")
                    break
            
            if not self._check_hourly_limit():
                if auto_resume:
                    logger.warning("Hourly rate limit reached. Waiting for reset...")
                    self._wait_for_hourly_reset()
                else:
                    logger.warning("Hourly rate limit reached. Stopping collection.")
                    break
            
            logger.info(f"Collecting train data for {current_date.strftime('%Y-%m-%d')} (Day {(current_date - start_date).days + 1}/{total_days})")
            
            try:
                daily_data = self._collect_daily_data(current_date)
                all_data.extend(daily_data)
                
                # Save raw data after each day
                if daily_data:
                    self._save_raw_data(daily_data, current_date, 'viaggiotreno')
                    logger.info(f"✅ Collected {len(daily_data)} records for {current_date.strftime('%Y-%m-%d')} (Total: {len(all_data)})")
                    self.consecutive_errors = 0  # Reset error counter on success
                else:
                    logger.warning(f"No data collected for {current_date.strftime('%Y-%m-%d')}")
                
                current_date += timedelta(days=1)
                
                # Longer pause between days to be respectful
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error collecting data for {current_date.strftime('%Y-%m-%d')}: {str(e)}")
                self.consecutive_errors += 1
                
                if auto_resume:
                    # Exponential backoff on errors
                    backoff_time = min(300, 30 * (2 ** self.consecutive_errors))
                    logger.info(f"Backing off for {backoff_time} seconds due to error")
                    time.sleep(backoff_time)
                    
                    # Skip this day and continue
                    current_date += timedelta(days=1)
                else:
                    logger.error("Auto-resume disabled, stopping on error")
                    break
        
        logger.info(f"Collection completed. Total records: {len(all_data)}")
        return all_data
    
    def _check_hourly_limit(self) -> bool:
        """Check if we're within hourly rate limits"""
        current_hour = datetime.now().hour
        
        # Reset counter if it's a new hour
        if current_hour != self.current_hour:
            self.requests_this_hour = 0
            self.current_hour = current_hour
        
        return self.requests_this_hour < self.requests_per_hour
    
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.requests_this_hour += 1
    
    def _collect_daily_data(self, date: datetime) -> List[Dict[Any, Any]]:
        """Collect train data for a specific date with error handling"""
        daily_data = []
        
        for i, station in enumerate(self.stations):
            if not self._check_hourly_limit():
                logger.warning("Hourly limit reached during daily collection")
                break
            
            try:
                logger.debug(f"Collecting data for station {station['name']} ({i+1}/{len(self.stations)})")
                
                station_data = self._get_station_departures(station['code'], date)
                
                for train in station_data:
                    train['station_code'] = station['code']
                    train['station_name'] = station['name']
                    train['collection_date'] = date.strftime('%Y-%m-%d')
                
                daily_data.extend(station_data)
                logger.debug(f"Collected {len(station_data)} trains from {station['name']}")
                
                # Rate limiting between stations
                self._rate_limit()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error for station {station['code']}: {str(e)}")
                # Don't break the loop, continue with next station
                time.sleep(10)  # Extra delay on request errors
                continue
            except Exception as e:
                logger.error(f"Unexpected error for station {station['code']}: {str(e)}")
                continue
        
        return daily_data
    
    def _get_station_departures(self, station_code: str, date: datetime) -> List[Dict[Any, Any]]:
        """Get departures for a specific station and date with robust error handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Format date for Viaggiotreno API
                formatted_date = date.strftime('%a %b %d %Y %H:%M:%S GMT+0100 (CET)')
                url = f"{self.base_url}/partenze/{station_code}/{formatted_date}"
                
                logger.debug(f"Requesting: {url}")
                
                response = self.session.get(url, timeout=15)
                
                # Handle specific HTTP errors
                if response.status_code == 429:
                    logger.warning("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    retry_count += 1
                    continue
                elif response.status_code == 404:
                    logger.warning(f"No data found for station {station_code} on {date.strftime('%Y-%m-%d')}")
                    return []
                elif response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code}, retrying...")
                    time.sleep(10 * (retry_count + 1))
                    retry_count += 1
                    continue
                
                response.raise_for_status()
                
                # Handle empty or invalid JSON
                try:
                    departures = response.json()
                    if not isinstance(departures, list):
                        logger.warning(f"Unexpected response format for {station_code}")
                        return []
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response for {station_code}")
                    return []
                
                processed_departures = []
                for departure in departures:
                    processed_departure = self._process_departure(departure, station_code, date)
                    if processed_departure:
                        processed_departures.append(processed_departure)
                
                return processed_departures
                
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout for {station_code}, retry {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(5 * (retry_count + 1))
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error for {station_code}, retry {retry_count + 1}/{max_retries}")
                retry_count += 1
                time.sleep(10 * (retry_count + 1))
            except Exception as e:
                logger.error(f"Unexpected error fetching departures for {station_code}: {str(e)}")
                retry_count += 1
                time.sleep(5)
        
        logger.error(f"Failed to get departures for {station_code} after {max_retries} retries")
        return []
    
    def _process_departure(self, departure: Dict, station_code: str, date: datetime) -> Dict[Any, Any]:
        """Process a single departure record with improved error handling"""
        try:
            # Extract key information with safe gets
            train_id = str(departure.get('numeroTreno', ''))
            category = departure.get('categoriaDescrizione', '')
            
            # Get scheduled and actual times
            scheduled_time = departure.get('orarioPartenza', '')
            actual_time = departure.get('orarioPartenzaReale', '')
            
            # Calculate delay with multiple fallback methods
            delay_minutes = 0
            delay_status = 'on_time'
            
            if actual_time and scheduled_time:
                try:
                    # Try to parse times and calculate delay
                    scheduled_dt = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {scheduled_time}", '%Y-%m-%d %H%M')
                    actual_dt = datetime.strptime(f"{date.strftime('%Y-%m-%d')} {actual_time}", '%Y-%m-%d %H%M')
                    delay_minutes = int((actual_dt - scheduled_dt).total_seconds() / 60)
                    
                    if delay_minutes > 5:
                        delay_status = 'delayed'
                    elif delay_minutes < -2:
                        delay_status = 'early'
                        
                except ValueError:
                    # Fallback to ritardo field
                    delay_minutes = departure.get('ritardo', 0)
                    if delay_minutes > 0:
                        delay_status = 'delayed'
            else:
                # Use ritardo field if times are not available
                delay_minutes = departure.get('ritardo', 0)
                if delay_minutes > 0:
                    delay_status = 'delayed'
            
            # Check for cancellation
            if departure.get('soppresso', False) or departure.get('cancellato', False):
                delay_status = 'cancelled'
            
            return {
                'train_id': train_id,
                'date': date.strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                'station_code': station_code,
                'scheduled_time': scheduled_time,
                'actual_time': actual_time,
                'delay_minutes': delay_minutes,
                'delay_status': delay_status,
                'train_category': category,
                'destination': departure.get('destinazione', ''),
                'platform': departure.get('binarioProgrammatoPartenzaDescrizione', ''),
                'is_cancelled': departure.get('soppresso', False),
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing departure {departure.get('numeroTreno', 'unknown')}: {str(e)}")
            return None
    
    def _save_raw_data(self, data: List[Dict], date: datetime, source: str):
        """Save raw data to file"""
        try:
            date_str = date.strftime('%Y-%m-%d')
            os.makedirs(f'data/raw/{source}/{date_str}', exist_ok=True)
            
            filename = f'data/raw/{source}/{date_str}/data_{datetime.now().strftime("%H%M%S")}.json'
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Saved raw data to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        return {
            'requests_this_hour': self.requests_this_hour,
            'hourly_limit': self.requests_per_hour,
            'remaining_requests': self.requests_per_hour - self.requests_this_hour,
            'current_hour': self.current_hour,
            'rate_limit_seconds': self.min_request_interval,
            'consecutive_errors': self.consecutive_errors,
            'max_consecutive_errors': self.max_consecutive_errors,
            'can_make_requests': self._check_hourly_limit() and self.consecutive_errors < self.max_consecutive_errors
        }
    
    def can_collect_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Check if we can collect data for the given date range"""
        total_days = (end_date - start_date).days + 1
        estimated_requests = len(self.stations) * total_days
        
        can_collect = (
            self._check_hourly_limit() and 
            self.consecutive_errors < self.max_consecutive_errors and
            estimated_requests <= (self.requests_per_hour - self.requests_this_hour)
        )
        
        reason = 'OK'
        if not self._check_hourly_limit():
            reason = 'Hourly limit reached'
        elif self.consecutive_errors >= self.max_consecutive_errors:
            reason = 'Too many consecutive errors'
        elif estimated_requests > (self.requests_per_hour - self.requests_this_hour):
            reason = 'Not enough remaining requests for this collection'
        
        return {
            'can_collect': can_collect,
            'estimated_requests': estimated_requests,
            'remaining_requests': self.requests_per_hour - self.requests_this_hour,
            'total_days': total_days,
            'reason': reason
        }
    
    def _wait_for_hourly_reset(self):
        """Wait for hourly API limit to reset (next hour)"""
        now = datetime.now()
        next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        wait_seconds = (next_hour - now).total_seconds()
        
        logger.info(f"Waiting {wait_seconds/60:.1f} minutes for hourly limit reset at {next_hour.strftime('%H:%M:%S')}")
        
        # Wait in chunks to allow for graceful interruption
        while wait_seconds > 0:
            chunk_size = min(600, wait_seconds)  # Wait in 10-minute chunks
            logger.info(f"Sleeping for {chunk_size/60:.1f} minutes... ({wait_seconds/60:.1f} minutes remaining)")
            time.sleep(chunk_size)
            wait_seconds -= chunk_size
            
            # Check if we're close to reset time
            if wait_seconds <= 60:
                logger.info("Hourly limit reset imminent, checking status...")
                break
        
        # Reset our counter
        self.requests_this_hour = 0
        self.current_hour = datetime.now().hour
        logger.info("Hourly limit has been reset, resuming collection...")
    
    def reset_error_counter(self):
        """Reset consecutive error counter (useful for manual recovery)"""
        self.consecutive_errors = 0
        logger.info("Error counter reset")
    
    def get_train_details(self, train_id: str, station_code: str) -> Dict[Any, Any]:
        """Get detailed information for a specific train with rate limiting"""
        try:
            if not self._check_hourly_limit():
                logger.warning("Hourly limit reached, cannot fetch train details")
                return {}
            
            url = f"{self.base_url}/andamentoTreno/{station_code}/{train_id}"
            
            self._rate_limit()
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 429:
                logger.warning("Rate limit hit while fetching train details")
                return {}
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching train details for {train_id}: {str(e)}")
            return {}