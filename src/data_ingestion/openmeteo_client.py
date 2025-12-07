"""
Open-Meteo API client for collecting weather data (scalable)
"""

import requests
import json
import time
import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple
from loguru import logger
import csv
from pathlib import Path


class OpenMeteoClient:
    """Client for Open-Meteo API to collect weather data at scale"""

    def __init__(self, stations_csv: str | None = None):
        self.base_url = "https://api.open-meteo.com/v1"
        self.session = requests.Session()

        # TODO: qui metterai le 324 stazioni lombarde

        if stations_csv:
            self.stations = self._load_stations_from_csv(stations_csv)
        else:       
            self.stations = [
                {'code': 'S01700', 'name': 'Roma Termini', 'lat': 41.9009, 'lon': 12.5024},
                {'code': 'S02430', 'name': 'Milano Centrale', 'lat': 45.4842, 'lon': 9.2044},
                {'code': 'S05042', 'name': 'Napoli Centrale', 'lat': 40.8518, 'lon': 14.2681},
                {'code': 'S06409', 'name': 'Firenze S.M.N.', 'lat': 43.7766, 'lon': 11.2480},
                {'code': 'S02593', 'name': 'Torino Porta Nuova', 'lat': 45.0619, 'lon': 7.6792},
                {'code': 'S05915', 'name': 'Bologna Centrale', 'lat': 44.5059, 'lon': 11.3426},
                {'code': 'S02667', 'name': 'Venezia S. Lucia', 'lat': 45.4408, 'lon': 12.3208},
                {'code': 'S08409', 'name': 'Bari Centrale', 'lat': 41.1171, 'lon': 16.8719}
            ]

    # ---------- PUBLIC API ----------

    def collect_data_streaming(
        self,
        start_date: datetime,
        end_date: datetime,
        chunk_days: int = 7,
        base_dir: str = "data",
        save_raw: bool = True,
    ) -> int:
        """
        Collect weather data in chunks and stream to disk (CSV + opzionale RAW).

        - NON accumula tutto in memoria
        - spezza l'intervallo [start_date, end_date] in chunk da `chunk_days`
        - salta i chunk già presenti su disco (auto-resume)

        Ritorna il numero totale di record meteo processati.
        """
        all_chunks = self._build_date_chunks(start_date.date(), end_date.date(), chunk_days)
        total_records = 0

        logger.info(f"Total stations: {len(self.stations)}")
        logger.info(f"Total chunks: {len(all_chunks)} (chunk_days={chunk_days})")

        for station in self.stations:
            logger.info(f"=== Station {station['code']} - {station['name']} ===")
            for chunk_start, chunk_end in all_chunks:
                # file CSV di output per questo station+chunk
                csv_path = self._get_curated_csv_path(
                    base_dir,
                    station["code"],
                    chunk_start,
                    chunk_end
                )

                # se il file esiste, assumiamo che questo chunk sia già stato fatto → auto-resume
                if os.path.exists(csv_path):
                    logger.info(f"SKIP chunk {chunk_start} → {chunk_end} (file exists: {csv_path})")
                    continue

                logger.info(f"Fetching {station['code']}  {chunk_start} → {chunk_end}")
                try:
                    data = self._get_station_weather(station, chunk_start, chunk_end)
                    if not data:
                        logger.warning(f"No data for {station['code']} on chunk {chunk_start} → {chunk_end}")
                        continue

                    # opzionale: salva RAW (una risposta per chunk)
                    if save_raw:
                        self._save_raw_data(data, station, chunk_start, chunk_end, base_dir)

                    # processa e scrivi CURATED (append CSV)
                    processed = self._process_weather_data(data, station)
                    self._append_curated_csv(processed, csv_path)

                    total_records += len(processed)
                    logger.info(f"Chunk {chunk_start} → {chunk_end}: {len(processed)} records")

                    # piccolo sleep difensivo per non abusare dell'API
                    time.sleep(0.2)

                except Exception as e:
                    logger.error(
                        f"Error collecting data for {station['code']} "
                        f"{chunk_start}→{chunk_end}: {e}"
                    )
                    # NON interrompe l'intero processo, passa al prossimo chunk
                    continue

        logger.info(f"TOTAL processed weather records: {total_records}")
        return total_records

    # ---------- INTERNAL HELPERS ----------
    def _load_stations_from_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Carica le stazioni da un file CSV con colonne:
        station_code,station_name,lat,lon

        Esempio riga:
        S02430,Milano Centrale,45.4842,9.2044
        """
        path = Path(csv_path)
        if not path.exists():
            logger.warning(f"Stations CSV not found: {csv_path}. Using fallback stations.")
            return self.stations if hasattr(self, "stations") else []

        stations: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    stations.append(
                        {
                            "code": row["station_code"],
                            "name": row["station_name"],
                            "lat": float(row["lat"]),
                            "lon": float(row["lon"]),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error parsing station row {row}: {e}")

        logger.info(f"Loaded {len(stations)} stations from {csv_path}")
        return stations

    def _build_date_chunks(
        self,
        start: date,
        end: date,
        chunk_days: int
    ) -> List[Tuple[date, date]]:
        """Spezzetta [start, end] in intervalli chiusi di chunk_days."""
        chunks = []
        current = start
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    def _request_with_retry(
        self,
        url: str,
        params: Dict[str, Any],
        timeout_sec: int = 30,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Esegue una GET con retry + backoff esponenziale.
        Gestisce rate limit (429) e temporanei down delle API.

        Ritorna il JSON se ok, altrimenti lancia eccezione dopo max_retries.
        """
        last_exc = None
        backoff = backoff_base

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=timeout_sec)
                status = resp.status_code

                # gestione esplicita rate limit
                if status == 429:
                    logger.warning(
                        f"429 Too Many Requests (attempt {attempt}/{max_retries}), "
                        f"sleep {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.warning(
                    f"HTTP error (attempt {attempt}/{max_retries}): {e} "
                    f"- sleep {backoff}s"
                )
                time.sleep(backoff)
                backoff *= 2

        raise RuntimeError(f"Max retries exceeded for URL {url}: {last_exc}")

    def _get_station_weather(
        self,
        station: Dict[str, Any],
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Get weather data for a specific station and date range (chunk)."""

        hourly_vars = [
            "temperature_2m",
            "precipitation",
            "wind_speed_10m",
            "weather_code",
            "visibility",
        ]

        params = {
            "latitude": station["lat"],
            "longitude": station["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(hourly_vars),  # stringa comma-separated
            "timezone": "Europe/Rome",
        }

        url = f"{self.base_url}/forecast"
        logger.debug(f"Request Open-Meteo: {url} {params}")

        data = self._request_with_retry(url, params)
        return data

    def _process_weather_data(self, data: Dict, station: Dict) -> List[Dict[Any, Any]]:
        """Process weather API response into structured records."""
        processed_data: List[Dict[str, Any]] = []

        try:
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            temperatures = hourly.get("temperature_2m", [])
            precipitations = hourly.get("precipitation", [])
            wind_speeds = hourly.get("wind_speed_10m", [])
            weather_codes = hourly.get("weather_code", [])
            visibility = hourly.get("visibility", [])

            for i, timestamp in enumerate(times):
                record = {
                    "station_code": station["code"],
                    "station_name": station["name"],
                    "latitude": station["lat"],
                    "longitude": station["lon"],
                    "timestamp": timestamp,  # string ISO, Postgres lo digerisce
                    "temperature": temperatures[i] if i < len(temperatures) else None,
                    "wind_speed": wind_speeds[i] if i < len(wind_speeds) else None,
                    "precip_mm": precipitations[i] if i < len(precipitations) else None,
                    "weather_code": weather_codes[i] if i < len(weather_codes) else None,
                    "visibility": visibility[i] if i < len(visibility) else None,
                }
                processed_data.append(record)

            return processed_data

        except Exception as e:
            logger.error(f"Error processing weather data: {str(e)}")
            return []

    # ---------- STORAGE HELPERS ----------

    def _get_curated_csv_path(
        self,
        base_dir: str,
        station_code: str,
        chunk_start: date,
        chunk_end: date,
    ) -> str:
        """
        Genera il path per il CSV curated di uno specifico chunk:
        data/curated/openmeteo/<station_code>/YYYY/weather_S02430_2025-01-01_2025-01-07.csv
        """
        year = chunk_start.year
        dir_path = os.path.join(base_dir, "curated", "openmeteo", station_code, str(year))
        os.makedirs(dir_path, exist_ok=True)

        filename = f"weather_{station_code}_{chunk_start}_{chunk_end}.csv"
        return os.path.join(dir_path, filename)

    def _append_curated_csv(
        self,
        records: List[Dict[str, Any]],
        csv_path: str
    ):
        """Appende i record curated a un CSV (creando header se non esiste)."""
        if not records:
            return

        file_exists = os.path.exists(csv_path)

        fieldnames = [
            "station_code",
            "station_name",
            "latitude",
            "longitude",
            "timestamp",
            "temperature",
            "wind_speed",
            "precip_mm",
            "weather_code",
            "visibility",
        ]

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in records:
                writer.writerow(r)

        logger.info(f"[CURATED] Wrote {len(records)} records to {csv_path}")

    def _save_raw_data(
        self,
        data: Dict[str, Any],
        station: Dict[str, Any],
        chunk_start: date,
        chunk_end: date,
        base_dir: str,
    ):
        """
        Save raw JSON for a station and date-chunk.
        """
        try:
            date_str = chunk_start.strftime("%Y-%m-%d")
            dir_path = os.path.join(base_dir, "raw", "openmeteo", station["code"])
            os.makedirs(dir_path, exist_ok=True)

            filename = f"raw_{station['code']}_{date_str}_{chunk_end}.json"
            file_path = os.path.join(dir_path, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"[RAW] Saved raw weather data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving raw weather data: {str(e)}")

    # ---------- WEATHER CODES MAPPING ----------

    def get_weather_codes_mapping(self) -> Dict[int, str]:
        """Get mapping of weather codes to descriptions."""
        return {
            0: 'Clear sky',
            1: 'Mainly clear',
            2: 'Partly cloudy',
            3: 'Overcast',
            45: 'Fog',
            48: 'Depositing rime fog',
            51: 'Light drizzle',
            53: 'Moderate drizzle',
            55: 'Dense drizzle',
            56: 'Light freezing drizzle',
            57: 'Dense freezing drizzle',
            61: 'Slight rain',
            63: 'Moderate rain',
            65: 'Heavy rain',
            66: 'Light freezing rain',
            67: 'Heavy freezing rain',
            71: 'Slight snow fall',
            73: 'Moderate snow fall',
            75: 'Heavy snow fall',
            77: 'Snow grains',
            80: 'Slight rain showers',
            81: 'Moderate rain showers',
            82: 'Violent rain showers',
            85: 'Slight snow showers',
            86: 'Heavy snow showers',
            95: 'Thunderstorm',
            96: 'Thunderstorm with slight hail',
            99: 'Thunderstorm with heavy hail'
        }
