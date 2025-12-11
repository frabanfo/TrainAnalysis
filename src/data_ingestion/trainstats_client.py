import requests
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from loguru import logger
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class TrainStatsClient:
    
    def __init__(self):
        self.base_url = "https://trainstats.altervista.org/speciali/stazioni/table.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.min_request_interval = 1.0
        self.last_request_time = 0

    
    def _rate_limit(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _format_date_for_api(self, date: datetime) -> str:
        return date.strftime('%d_%m_%Y')
    
    def get_station_data(self, station_name: str, station_code: str ,date: datetime) -> List[Dict[Any, Any]]:
        try:
            self._rate_limit()
            
            date_str = self._format_date_for_api(date)
            
            params = {
                'data': date_str,
                'n': station_name
            }
            
            logger.debug(f"Requesting data for {station_name} on {date.strftime('%Y-%m-%d')}")
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            html_content = response.text
            
            table_match = re.search(r"var table = \[(.*?)\];", html_content, re.DOTALL)
            
            if not table_match:
                logger.warning(f"No data table found for {station_name} on {date.strftime('%Y-%m-%d')}")
                return []
            
            table_content = table_match.group(1)
            
            entries = []
            current_entry = ""
            in_quotes = False
            
            for char in table_content:
                if char == "'" and not in_quotes:
                    in_quotes = True
                    current_entry = ""
                elif char == "'" and in_quotes:
                    in_quotes = False
                    if current_entry.strip():
                        entries.append(current_entry)
                elif in_quotes:
                    current_entry += char
            
            if not entries:
                logger.warning(f"No train entries found for {station_name} on {date.strftime('%Y-%m-%d')}")
                return []
            
            # Skip header row (first entry)
            train_entries = entries[1:] if len(entries) > 1 else []
            
            processed_trains = []
            for entry in train_entries:
                try:
                    train_data = self._parse_train_entry(entry, station_code, date)
                    if train_data:
                        processed_trains.append(train_data)
                except Exception as e:
                    logger.warning(f"Error parsing train entry '{entry[:50]}...': {str(e)}")
                    continue
            
            logger.info(f"Collected {len(processed_trains)} trains from {station_name} for {date.strftime('%Y-%m-%d')}")
            return processed_trains
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {station_name} on {date.strftime('%Y-%m-%d')}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error for {station_name} on {date.strftime('%Y-%m-%d')}: {str(e)}")
            return []
    
    def _parse_train_entry(self, entry: str, station_code: str, date: datetime) -> Optional[Dict[Any, Any]]:
        try:
            fields = entry.split(',')
            
            if len(fields) < 9:
                return None
            
            categoria = fields[0].strip()
            num_treno = fields[1].strip()
            tratta = fields[2].strip() if len(fields) > 2 else ""
            arrivo_prog = fields[3].strip() if len(fields) > 3 else ""
            arrivo_reale = fields[4].strip() if len(fields) > 4 else ""
            rit_arrivo = fields[5].strip() if len(fields) > 5 else "0"
            partenza_prog = fields[6].strip() if len(fields) > 6 else ""
            partenza_reale = fields[7].strip() if len(fields) > 7 else ""
            rit_partenza = fields[8].strip() if len(fields) > 8 else "0"
            
            try:
                delay_arrival = int(rit_arrivo) if rit_arrivo and rit_arrivo.isdigit() else 0
            except:
                delay_arrival = 0
                
            try:
                delay_departure = int(rit_partenza) if rit_partenza and rit_partenza.isdigit() else 0
            except:
                delay_departure = 0
            
            delay_minutes = delay_departure if delay_departure != 0 else delay_arrival
            
            # Determine delay status
            if 'X' in partenza_reale or 'Non rilevato' in partenza_reale:
                delay_status = 'cancelled'
            elif delay_minutes > 5:
                delay_status = 'delayed'
            elif delay_minutes < -2:
                delay_status = 'early'
            else:
                delay_status = 'on_time'
            
            scheduled_time = self._parse_time_field(partenza_prog, date)
            actual_time = self._parse_time_field(partenza_reale, date) if 'X' not in partenza_reale and 'Non rilevato' not in partenza_reale else None
            
            destination = self._extract_destination(tratta)
            
            return {
                'train_id': num_treno,
                'timestamp': scheduled_time if scheduled_time else datetime.now(),
                'station_code': station_code,
                'scheduled_time': scheduled_time,
                'actual_time': actual_time,
                'delay_minutes': delay_minutes,
                'train_category': categoria,
                'route': tratta,
                'delay_status': delay_status,
                'destination': destination,
                'is_cancelled': delay_status == 'cancelled',
            }
            
        except Exception as e:
            logger.error(f"Error parsing train entry: {str(e)}")
            return None
    
    def _parse_time_field(self, time_str: str, date: datetime) -> Optional[datetime]:
        if not time_str or time_str in ['X', 'Non rilevato', '']:
            return None
        
        try:
            # Format: "HH:MM:SS - DD/MM"
            if ' - ' in time_str:
                time_part = time_str.split(' - ')[0]
            else:
                time_part = time_str
            
            # Parse time
            if ':' in time_part:
                time_parts = time_part.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1])
                
                return date.replace(hour=hour, minute=minute, second=0, microsecond=0)
        except:
            pass
        
        return None
    
    def _extract_destination(self, route: str) -> str:
        if not route:
            return ""
        
        try:
            if '<br>' in route:
                parts = route.split('<br>')
                if len(parts) > 1:
                    dest_part = parts[1]
                    if '(' in dest_part:
                        return dest_part.split('(')[0].strip()
                    return dest_part.strip()
            
            return route.strip()
        except:
            return route
    
    def save_raw_data(self, data: List[Dict], date: datetime, station: str):
        try:
            import os
            date_str = date.strftime('%Y-%m-%d')
            os.makedirs(f'data/raw/trainstats/{date_str}', exist_ok=True)
            
            filename = f'data/raw/trainstats/{date_str}/{station}_{datetime.now().strftime("%H%M%S")}.json'
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
            logger.info(f"Saved raw TrainStats data to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving raw data: {str(e)}")
