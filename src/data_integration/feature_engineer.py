import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger


class FeatureEngineer:
    def __init__(self):
        self.feature_config = {
            'enable_temporal_features': True,
            'enable_weather_features': True,
            'enable_delay_features': True,
            'enable_categorical_features': True,
            'rush_hour_ranges': [(7, 9), (17, 19)],  # Morning and evening rush hours
            'weekend_days': [5, 6],  # Saturday and Sunday (0=Monday)
        }
    
    def engineer_features(self, integrated_data: pd.DataFrame) -> pd.DataFrame:
        if integrated_data.empty:
            logger.warning("No data provided for feature engineering")
            return integrated_data
        
        logger.info(f"Engineering features for {len(integrated_data)} records")
        
        df = integrated_data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            if self.feature_config['enable_temporal_features']:
                df = self._add_temporal_features(df)
            
            if self.feature_config['enable_weather_features']:
                df = self._add_weather_features(df)
            
            if self.feature_config['enable_delay_features']:
                df = self._add_delay_features(df)
            
            if self.feature_config['enable_categorical_features']:
                df = self._add_categorical_features(df)
            
            logger.info(f"Feature engineering completed: {len(df.columns)} total columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            return integrated_data  # Return original data on failure
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding temporal features")
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found for temporal features")
            return df
        
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        df['is_weekend'] = df['day_of_week'].isin(self.feature_config['weekend_days'])
        
        # Rush hour indicator
        df['is_rush_hour'] = False
        for start_hour, end_hour in self.feature_config['rush_hour_ranges']:
            df.loc[
                (df['hour_of_day'] >= start_hour) & (df['hour_of_day'] < end_hour),
                'is_rush_hour'
            ] = True
        
        # Time period categories
        if df['hour_of_day'].nunique() > 1:
            try:
                df['time_period'] = pd.cut(
                    df['hour_of_day'],
                    bins=[0, 6, 12, 18, 24],
                    labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                    include_lowest=True
                )
            except ValueError as e:
                logger.warning(f"Could not create time period categories: {e}")
                df['time_period'] = 'Unknown'
        else:
            df['time_period'] = 'Unknown'
        
        # Season (Northern Hemisphere)
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        logger.debug("Temporal features added successfully")
        return df
    
    def _add_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding weather features")
        
        # Temperature categories
        if 'temperature' in df.columns and df['temperature'].notna().nunique() > 1:
            try:
                df['temp_category'] = pd.cut(
                    df['temperature'],
                    bins=[-np.inf, 0, 10, 20, 30, np.inf],
                    labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
                )
            except ValueError as e:
                logger.warning(f"Could not create temperature categories: {e}")
                df['temp_category'] = 'Unknown'
            
            # Temperature extremes
            df['is_extreme_cold'] = df['temperature'] < -10
            df['is_extreme_hot'] = df['temperature'] > 35
            df['is_freezing'] = df['temperature'] <= 0
        
        # Precipitation features
        if 'precip_mm' in df.columns:
            df['is_raining'] = df['precip_mm'] > 0
            
            # Rain intensity categories
            precip_values = df['precip_mm'].fillna(0)
            if precip_values.nunique() > 1:
                try:
                    df['rain_intensity'] = pd.cut(
                        precip_values,
                        bins=[0, 0.1, 2.5, 10, 50, np.inf],
                        labels=['None', 'Light', 'Moderate', 'Heavy', 'Extreme'],
                        include_lowest=True
                    )
                except ValueError as e:
                    logger.warning(f"Could not create rain intensity categories: {e}")
                    df['rain_intensity'] = 'None'
            else:
                df['rain_intensity'] = 'None'
            
            # Heavy rain indicator
            df['is_heavy_rain'] = df['precip_mm'] > 10
        
        # Wind features
        if 'wind_speed' in df.columns:
            wind_values = df['wind_speed'].fillna(0)
            if wind_values.nunique() > 1:
                try:
                    df['wind_category'] = pd.cut(
                        wind_values,
                        bins=[0, 5, 15, 25, 35, np.inf],
                        labels=['Calm', 'Light', 'Moderate', 'Strong', 'Severe'],
                        include_lowest=True
                    )
                except ValueError as e:
                    logger.warning(f"Could not create wind categories: {e}")
                    df['wind_category'] = 'Calm'
            else:
                df['wind_category'] = 'Calm'
            
            # High wind indicator
            df['is_high_wind'] = df['wind_speed'] > 25
        
        # Weather severity score (0-1)
        weather_severity = 0
        severity_factors = 0
        
        if 'temperature' in df.columns:
            # Temperature severity (extreme temps increase severity)
            temp_severity = np.where(
                (df['temperature'] < -5) | (df['temperature'] > 35),
                0.3, 0
            )
            weather_severity += temp_severity
            severity_factors += 1
        
        if 'precip_mm' in df.columns:
            # Precipitation severity
            precip_severity = np.minimum(df['precip_mm'].fillna(0) / 20, 0.4)  # Max 0.4 for precip
            weather_severity += precip_severity
            severity_factors += 1
        
        if 'wind_speed' in df.columns:
            # Wind severity
            wind_severity = np.minimum(df['wind_speed'].fillna(0) / 50, 0.3)  # Max 0.3 for wind
            weather_severity += wind_severity
            severity_factors += 1
        
        if severity_factors > 0:
            df['weather_severity_score'] = np.minimum(weather_severity, 1.0)
        else:
            df['weather_severity_score'] = 0
        
        # Adverse weather conditions
        df['adverse_weather'] = (
            df.get('is_extreme_cold', False) |
            df.get('is_extreme_hot', False) |
            df.get('is_heavy_rain', False) |
            df.get('is_high_wind', False)
        )
        
        logger.debug("Weather features added successfully")
        return df
    
    def _add_delay_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding delay features")
        
        if 'delay_minutes' in df.columns:
            # Delay categories
            delay_values = df['delay_minutes'].fillna(0)
            if delay_values.nunique() > 1:
                try:
                    df['delay_category'] = pd.cut(
                        delay_values,
                        bins=[-np.inf, -5, 5, 15, 30, 60, np.inf],
                        labels=['Early', 'On_Time', 'Minor_Delay', 'Moderate_Delay', 'Major_Delay', 'Severe_Delay']
                    )
                except ValueError as e:
                    logger.warning(f"Could not create delay categories: {e}")
                    df['delay_category'] = 'On_Time'
            else:
                df['delay_category'] = 'On_Time'
            
            # Binary delay indicators
            df['is_delayed'] = df['delay_minutes'] > 5
            df['is_significantly_delayed'] = df['delay_minutes'] > 15
            df['is_severely_delayed'] = df['delay_minutes'] > 60
            df['is_early'] = df['delay_minutes'] < -5
            
            # Delay severity score (0-1)
            df['delay_severity_score'] = np.minimum(
                np.maximum(df['delay_minutes'].fillna(0) / 120, 0), 1  # Normalize to 2 hours max
            )
        
        # Cancellation features
        if 'is_cancelled' in df.columns:
            df['service_disruption'] = df['is_cancelled']
        else:
            df['service_disruption'] = False
        
        # Delay status encoding
        if 'delay_status' in df.columns:
            df['delay_status_encoded'] = df['delay_status'].map({
                'on_time': 0,
                'early': -1,
                'delayed': 1,
                'cancelled': 2
            }).fillna(0)
        
        logger.debug("Delay features added successfully")
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Adding categorical features")
        
        # Train category features
        if 'train_category' in df.columns:
            # High-speed vs regular trains
            high_speed_categories = ['ICE', 'TGV', 'AVE', 'Frecciarossa', 'Shinkansen']
            df['is_high_speed'] = df['train_category'].isin(high_speed_categories)
            
            # Regional vs long-distance
            regional_categories = ['RE', 'RB', 'S', 'Regional']
            df['is_regional'] = df['train_category'].isin(regional_categories)
        
        # Station importance (based on traffic volume)
        if 'station_code' in df.columns:
            station_counts = df['station_code'].value_counts()
            df['station_traffic_rank'] = df['station_code'].map(station_counts)
            
            # Categorize stations by traffic
            if station_counts.nunique() > 1:
                try:
                    traffic_quantiles = station_counts.quantile([0.33, 0.67])
                    df['station_importance'] = pd.cut(
                        df['station_traffic_rank'],
                        bins=[0, traffic_quantiles.iloc[0], traffic_quantiles.iloc[1], np.inf],
                        labels=['Low_Traffic', 'Medium_Traffic', 'High_Traffic']
                    )
                except ValueError as e:
                    logger.warning(f"Could not create station importance categories: {e}")
                    df['station_importance'] = 'Medium_Traffic'
            else:
                df['station_importance'] = 'Medium_Traffic'
        
        # Route complexity (simple heuristic based on route string length)
        if 'route' in df.columns:
            df['route_complexity'] = df['route'].str.len().fillna(0)
            df['is_complex_route'] = df['route_complexity'] > df['route_complexity'].median()
        
        # Destination distance (heuristic based on destination name)
        if 'destination' in df.columns:
            # Major cities (longer distances)
            major_cities = ['Berlin', 'Munich', 'Hamburg', 'Frankfurt', 'Cologne', 'Stuttgart']
            df['is_major_destination'] = df['destination'].str.contains('|'.join(major_cities), na=False)
        
        logger.debug("Categorical features added successfully")
        return df
    

    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:        
        if df.empty:
            return {'total_features': 0, 'feature_categories': {}}
        
        feature_categories = {
            'temporal': [col for col in df.columns if any(term in col.lower() 
                        for term in ['hour', 'day', 'month', 'year', 'weekend', 'rush', 'season', 'time'])],
            'weather': [col for col in df.columns if any(term in col.lower() 
                       for term in ['temp', 'rain', 'wind', 'weather', 'precip', 'adverse'])],
            'delay': [col for col in df.columns if any(term in col.lower() 
                     for term in ['delay', 'early', 'cancelled', 'disruption'])],
            'categorical': [col for col in df.columns if any(term in col.lower() 
                           for term in ['category', 'importance', 'traffic', 'speed', 'regional'])]
        }
        
        return {
            'total_features': len(df.columns),
            'feature_categories': {cat: len(features) for cat, features in feature_categories.items()},
            'feature_list': {cat: features for cat, features in feature_categories.items()},
            'data_types': df.dtypes.value_counts().to_dict()
        }