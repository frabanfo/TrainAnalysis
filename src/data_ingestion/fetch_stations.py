"""
Fetch train stations from Lombardia region using Viaggiatreno API
"""

import requests
import pandas as pd
from typing import List, Dict, Any
from loguru import logger
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.db_manager import DatabaseManager

class StationsFetcher:
    
    def __init__(self):
        self.api_url = "http://www.viaggiatreno.it/infomobilita/resteasy/viaggiatreno/elencoStazioni/1"
        self.db_manager = DatabaseManager()
        
    def fetch_stations(self) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Fetching stations from: {self.api_url}")
            
            response = requests.get(self.api_url, timeout=30)
            response.raise_for_status()
            
            stations_data = response.json()
            
            if not stations_data:
                logger.warning("No station data received from API")
                return []
            
            stations = []
            logger.info(f"Processing {len(stations_data)} station entries")
            
            for station_data in stations_data:
                try:
                    station_code = station_data.get('codiceStazione', '')
                    station_name = station_data.get('localita', {}).get('nomeLungo', '')
                    latitude = station_data.get('lat')
                    longitude = station_data.get('lon')
                    
                    if not station_code:
                        continue
                    
                    if latitude and longitude:
                        lat_float = float(latitude)
                        lon_float = float(longitude)
                        is_lombardia_coords = (44.5 <= lat_float <= 46.5 and 8.5 <= lon_float <= 11.5)
                    
                    if not is_lombardia_coords:
                        logger.warning(f"for this {station_data} cords are out of bound")

                    
                    if is_lombardia_coords:
                        stations.append({
                            'station_code': station_code,
                            'station_name': station_name,
                            'latitude': float(latitude) if latitude else None,
                            'longitude': float(longitude) if longitude else None
                        })
                        
                except Exception as e:
                    logger.warning(f"Error parsing station data {station_data}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(stations)} Lombardia stations")
            return stations
            
        except requests.RequestException as e:
            logger.error(f"Error fetching stations from API: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching stations: {str(e)}")
            return []
    
    def store_stations(self, stations: List[Dict[str, Any]]) -> bool:
        """Store stations in database using batch UPSERT"""
        if not stations:
            logger.warning("No stations to store")
            return False

        try:
            logger.info(f"Storing {len(stations)} stations in database (batch mode)")

            upsert_sql = """
            INSERT INTO stations (station_code, station_name, latitude, longitude)
            VALUES (:station_code, :station_name, :latitude, :longitude)
            ON CONFLICT (station_code)
            DO UPDATE SET
                station_name = EXCLUDED.station_name,
                latitude = COALESCE(EXCLUDED.latitude, stations.latitude),
                longitude = COALESCE(EXCLUDED.longitude, stations.longitude)
            """

            from sqlalchemy import text

            with self.db_manager.engine.begin() as conn:
                conn.execute(text(upsert_sql), stations) 

            logger.info(f"Successfully stored {len(stations)} stations")
            return True

        except Exception as e:
            logger.error(f"Error storing stations: {str(e)}")
            return False
    
    def get_stored_stations(self) -> pd.DataFrame:
        """Get all stored Lombardia stations"""
        query = """
        SELECT station_code, station_name, latitude, longitude, created_at
        FROM stations
        ORDER BY station_name
        """
        
        return self.db_manager.execute_query(query)
    
    def run(self) -> bool:
        logger.info("Starting Lombardia stations fetch process")
        
        stations = self.fetch_stations()
        
        if not stations:
            logger.error("No stations fetched from API")
            return False
        
        success = self.store_stations(stations)
        
        if success:
            return True
        else:
            logger.error("Failed to store stations in database")
            return False

def main():
    try:
        fetcher = StationsFetcher()
        success = fetcher.run()
        
        if success:
            logger.info("Lombardia stations fetch completed successfully")
        else:
            logger.error("Lombardia stations fetch failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()