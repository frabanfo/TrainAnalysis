import requests
import json
import time
import os
from datetime import datetime, date
from typing import List, Dict, Any, Tuple
from loguru import logger
import csv


class OpenMeteoClient:
    """Client per Open-Meteo (uso con worker Dramatiq, 1 stazione + 1 chunk per volta)."""

    def __init__(self, base_url: str | None = None):
        # QUI metti la base URL della Historical API (dai docs che stai usando)
        # Esempio tipico: "https://archive-api.open-meteo.com/v1"
        self.base_url = base_url or "https://archive-api.open-meteo.com/v1"
        self.session = requests.Session()

    def fetch_station_chunk(
        self,
        station: Dict[str, Any],
        chunk_start: date,
        chunk_end: date,
        base_dir: str = "data",
        save_raw: bool = False,
    ) -> int:

        csv_path = self._get_curated_csv_path(
            base_dir, station["code"], chunk_start, chunk_end
        )

        # Idempotenza / auto-resume
        if os.path.exists(csv_path):
            logger.info(
                f"[SKIP] {station['code']} {chunk_start}→{chunk_end} (file esiste: {csv_path})"
            )
            return 0

        logger.info(
            f"[FETCH] {station['code']} {station['name']} {chunk_start}→{chunk_end}"
        )

        data = self._get_station_weather(station, chunk_start, chunk_end)
        if not data:
            logger.warning(
                f"[NO DATA] {station['code']} {chunk_start}→{chunk_end} (nessun dato)"
            )
            return 0

        if save_raw:
            self._save_raw_data(data, station, chunk_start, chunk_end, base_dir)

        processed = self._process_weather_data(data, station)
        self._append_curated_csv(processed, csv_path)

        logger.info(
            f"[DONE] {station['code']} {chunk_start}→{chunk_end} -> {len(processed)} record"
        )
        return len(processed)

    def _get_station_weather(
        self,
        station: Dict[str, Any],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Any]:

        hourly_vars = [
            "temperature_2m",
            "precipitation",
            "windspeed_10m",   # ATTENZIONE ai nomi esatti dai docs
            "weathercode",
        ]

        params = {
            "latitude": station["lat"],
            "longitude": station["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(hourly_vars),
            "timezone": "Europe/Rome",
        }

        # Path da adattare ai docs (es. "/archive", "/era5" ecc.)
        url = f"{self.base_url}/archive"

        logger.debug(f"Request Open-Meteo: {url} {params}")
        data = self._request_with_retry(url, params)
        return data

    def _request_with_retry(
        self,
        url: str,
        params: Dict[str, Any],
        timeout_sec: int = 30,
        max_retries: int = 5,
        backoff_base: float = 2.0,
    ) -> Dict[str, Any]:
        """GET con retry + backoff esponenziale, gestisce rate-limit e problemi temporanei."""
        last_exc = None
        backoff = backoff_base

        for attempt in range(1, max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=timeout_sec)
                status = resp.status_code

                # 4xx non-429 -> errore cliente, non ha senso ritentare
                if 400 <= status < 500 and status != 429:
                    body = resp.text[:200] if resp.text else "<no body>"
                    logger.error(
                        f"Client error {status} for URL {resp.url}. "
                        f"Body (first 200 chars): {body}"
                    )
                    resp.raise_for_status()

                # rate limit esplicito
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

    # ---------- PROCESSING & STORAGE ----------

    def _process_weather_data(self, data: Dict, station: Dict) -> List[Dict[str, Any]]:
        processed: List[Dict[str, Any]] = []

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            logger.warning(
                f"No hourly data returned for station {station['code']} ({station['name']})"
            )
            return []

        temps = hourly.get("temperature_2m", [])
        precs = hourly.get("precipitation", [])
        winds = hourly.get("windspeed_10m", [])
        wcodes = hourly.get("weathercode", [])

        for i, ts in enumerate(times):
            record = {
                "station_code": station["code"],
                "station_name": station["name"],
                "latitude": station["lat"],
                "longitude": station["lon"],
                "timestamp": ts,
                "temperature": temps[i] if i < len(temps) else None,
                "wind_speed": winds[i] if i < len(winds) else None,
                "precip_mm": precs[i] if i < len(precs) else None,
                "weather_code": wcodes[i] if i < len(wcodes) else None,
            }
            processed.append(record)

        return processed

    def _get_curated_csv_path(
        self,
        base_dir: str,
        station_code: str,
        chunk_start: date,
        chunk_end: date,
    ) -> str:
        year = chunk_start.year
        dir_path = os.path.join(
            base_dir, "curated", "openmeteo", station_code, str(year)
        )
        os.makedirs(dir_path, exist_ok=True)
        filename = f"weather_{station_code}_{chunk_start}_{chunk_end}.csv"
        return os.path.join(dir_path, filename)

    def _append_curated_csv(
        self,
        records: List[Dict[str, Any]],
        csv_path: str
    ):
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
        ]

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for r in records:
                writer.writerow(r)

    def _save_raw_data(
        self,
        data: Dict[str, Any],
        station: Dict[str, Any],
        chunk_start: date,
        chunk_end: date,
        base_dir: str,
    ):
        try:
            dir_path = os.path.join(base_dir, "raw", "openmeteo", station["code"])
            os.makedirs(dir_path, exist_ok=True)
            filename = f"raw_{station['code']}_{chunk_start}_{chunk_end}.json"
            file_path = os.path.join(dir_path, filename)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving raw weather data: {e}")
