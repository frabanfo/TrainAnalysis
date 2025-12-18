#!/usr/bin/env python3
"""
Exploratory Data Analysis for Railway Weather Correlation Project
================================================================

This script performs comprehensive EDA to address the research questions:
1. Primary: What is the correlation between meteorological conditions and railway delay patterns in Lombardy?
2. Secondary: How do delays vary across temporal, spatial, and operational dimensions?

Author: Railway Analysis Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Database connection
import os
import sys
sys.path.append('src')
from database.db_manager import DatabaseManager

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class RailwayWeatherEDA:
    """Focused EDA for Railway-Weather Correlation Analysis"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data = {}
        self.figures_dir = "eda_figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def load_data(self):
        """Load all necessary data from database"""
        print("Loading data from database...")
        
        # Load integrated data with station names
        integrated_query = """
        SELECT 
            i.train_id, i.timestamp, i.station_code, s.station_name,
            i.delay_minutes, i.temperature, i.wind_speed, i.precip_mm, 
            i.weather_code, i.train_category, i.delay_status, 
            i.hour_of_day, i.day_of_week, i.is_weekend, i.is_rush_hour, 
            i.is_raining, i.rain_intensity, i.wind_category, i.is_delayed
        FROM train_weather_integrated i
        LEFT JOIN stations s ON i.station_code = s.station_code
        ORDER BY i.timestamp
        """
        
        # Load raw train data
        train_query = """
        SELECT 
            train_id, timestamp, station_code, delay_minutes, train_category,
            delay_status, is_cancelled
        FROM trains
        ORDER BY timestamp
        """
        
        # Load raw weather data
        weather_query = """
        SELECT 
            station_code, timestamp, temperature, wind_speed, precip_mm, weather_code
        FROM weather
        ORDER BY timestamp
        """
        
        # Load station data
        station_query = """
        SELECT station_code, station_name, latitude, longitude
        FROM stations
        """
        
        # Load data quality metrics
        dq_query = """
        SELECT table_name, metric_name, metric_value, timestamp, details
        FROM data_quality_metrics
        ORDER BY timestamp DESC
        """
        
        self.data['integrated'] = self.db_manager.execute_query(integrated_query)
        self.data['trains'] = self.db_manager.execute_query(train_query)
        self.data['weather'] = self.db_manager.execute_query(weather_query)
        self.data['stations'] = self.db_manager.execute_query(station_query)
        self.data['dq_metrics'] = self.db_manager.execute_query(dq_query)
        
        # Convert timestamps
        for key in ['integrated', 'trains', 'weather']:
            if not self.data[key].empty:
                self.data[key]['timestamp'] = pd.to_datetime(self.data[key]['timestamp'])
        
        if not self.data['dq_metrics'].empty:
            self.data['dq_metrics']['timestamp'] = pd.to_datetime(self.data['dq_metrics']['timestamp'])
        
        print(f"Data loaded successfully:")
        for key, df in self.data.items():
            print(f"  - {key}: {len(df)} records")
    
    def data_overview(self):
        """Generate data overview and basic statistics"""
        print("\n" + "="*60)
        print("DATA OVERVIEW")
        print("="*60)
        
        if self.data['integrated'].empty:
            print("No integrated data available for analysis")
            return
        
        df = self.data['integrated']
        
        print(f"Dataset Shape: {df.shape}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Number of Stations: {df['station_code'].nunique()}")
        print(f"Number of Unique Trains: {df['train_id'].nunique()}")
        
        # Data quality summary
        weather_match_rate = (df['temperature'].notna()).mean() * 100
        print(f"\nData Quality Summary:")
        print(f"Weather Integration Success: {weather_match_rate:.1f}%")
        print(f"Complete Records (weather + delay): {len(df.dropna(subset=['delay_minutes', 'temperature'])):,}")
        
        # Basic statistics for key variables
        print("\nKey Variable Statistics:")
        numeric_cols = ['delay_minutes', 'temperature', 'wind_speed', 'precip_mm']
        available_cols = [col for col in numeric_cols if col in df.columns]
        if available_cols:
            print(df[available_cols].describe().round(2))
    
    def plot_weather_impact_analysis(self):
        """Analyze the impact of bad weather on train delays - MAIN FOCUS"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        complete_data = df.dropna(subset=['delay_minutes', 'temperature', 'precip_mm'])
        
        if complete_data.empty:
            print("No complete weather-delay data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Rain Impact - The key relationship
        rain_comparison = complete_data.groupby('is_raining').agg({
            'delay_minutes': ['mean', 'std', 'count']
        }).round(2)
        rain_comparison.columns = ['mean_delay', 'std_delay', 'count']
        
        rain_labels = ['Clear Weather', 'Rainy Weather']
        colors = ['lightblue', 'darkblue']
        bars = axes[0,0].bar(range(len(rain_comparison)), rain_comparison['mean_delay'], 
                            yerr=rain_comparison['std_delay'], capsize=8,
                            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, mean_val, count) in enumerate(zip(bars, rain_comparison['mean_delay'], rain_comparison['count'])):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                          f'{mean_val:.1f} min\n(n={count:,})', ha='center', va='bottom', fontweight='bold')
        
        axes[0,0].set_title('Impact of Rain on Railway Delays', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('Average Delay (minutes)', fontsize=12)
        axes[0,0].set_xticks(range(len(rain_labels)))
        axes[0,0].set_xticklabels(rain_labels, fontsize=12)
        axes[0,0].grid(axis='y', alpha=0.3)
        
        # Calculate and show percentage increase
        if len(rain_comparison) == 2:
            increase = ((rain_comparison.iloc[1]['mean_delay'] - rain_comparison.iloc[0]['mean_delay']) / 
                       rain_comparison.iloc[0]['mean_delay']) * 100
            axes[0,0].text(0.5, 0.95, f'Delay increase: +{increase:.1f}%', 
                          transform=axes[0,0].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                          fontsize=11, fontweight='bold')
        
        # 2. Precipitation intensity vs delays
        complete_data['precip_category'] = pd.cut(complete_data['precip_mm'], 
                                                 bins=[-0.1, 0, 0.5, 2, 10], 
                                                 labels=['None', 'Light', 'Moderate', 'Heavy'])
        
        precip_delays = complete_data.groupby('precip_category')['delay_minutes'].agg(['mean', 'count'])
        precip_delays = precip_delays[precip_delays['count'] >= 100]  # Filter for significance
        
        if not precip_delays.empty:
            axes[0,1].bar(range(len(precip_delays)), precip_delays['mean'], 
                         color=['lightgreen', 'yellow', 'orange', 'red'][:len(precip_delays)], 
                         alpha=0.8, edgecolor='black')
            axes[0,1].set_title('Delays by Precipitation Intensity', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Average Delay (minutes)', fontsize=12)
            axes[0,1].set_xticks(range(len(precip_delays)))
            axes[0,1].set_xticklabels(precip_delays.index, fontsize=12)
            axes[0,1].grid(axis='y', alpha=0.3)
        
        # 3. Temperature extremes impact
        complete_data['temp_category'] = pd.cut(complete_data['temperature'], 
                                               bins=[-10, 0, 5, 15, 25, 40], 
                                               labels=['Very Cold', 'Cold', 'Cool', 'Mild', 'Hot'])
        
        temp_delays = complete_data.groupby('temp_category')['delay_minutes'].agg(['mean', 'count'])
        temp_delays = temp_delays[temp_delays['count'] >= 100]
        
        if not temp_delays.empty:
            temp_colors = ['darkblue', 'blue', 'lightblue', 'orange', 'red'][:len(temp_delays)]
            axes[1,0].bar(range(len(temp_delays)), temp_delays['mean'], 
                         color=temp_colors, alpha=0.8, edgecolor='black')
            axes[1,0].set_title('Delays by Temperature Category', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('Average Delay (minutes)', fontsize=12)
            axes[1,0].set_xticks(range(len(temp_delays)))
            axes[1,0].set_xticklabels(temp_delays.index, rotation=45, fontsize=12)
            axes[1,0].grid(axis='y', alpha=0.3)
        
        # 4. Combined weather conditions (rain + temperature)
        complete_data['weather_condition'] = complete_data.apply(lambda x: 
            'Rain + Cold' if x['is_raining'] and x['temperature'] < 5 else
            'Rain + Mild' if x['is_raining'] and x['temperature'] >= 5 else
            'Clear + Cold' if not x['is_raining'] and x['temperature'] < 5 else
            'Clear + Mild', axis=1)
        
        weather_delays = complete_data.groupby('weather_condition')['delay_minutes'].agg(['mean', 'count'])
        weather_delays = weather_delays[weather_delays['count'] >= 50]
        
        if not weather_delays.empty:
            condition_colors = {'Rain + Cold': 'darkblue', 'Rain + Mild': 'blue', 
                               'Clear + Cold': 'lightblue', 'Clear + Mild': 'lightgreen'}
            colors = [condition_colors.get(cond, 'gray') for cond in weather_delays.index]
            
            axes[1,1].bar(range(len(weather_delays)), weather_delays['mean'], 
                         color=colors, alpha=0.8, edgecolor='black')
            axes[1,1].set_title('Delays by Combined Weather Conditions', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('Average Delay (minutes)', fontsize=12)
            axes[1,1].set_xticks(range(len(weather_delays)))
            axes[1,1].set_xticklabels(weather_delays.index, rotation=45, fontsize=10)
            axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/01_weather_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_weather_patterns(self):
        """Analyze temporal patterns with weather correlation"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Rush hour impact with weather
        if 'is_rush_hour' in df.columns and 'is_raining' in df.columns:
            rush_weather = df.groupby(['is_rush_hour', 'is_raining'])['delay_minutes'].mean().unstack()
            
            if not rush_weather.empty:
                rush_weather.plot(kind='bar', ax=axes[0,0], color=['lightblue', 'darkblue'], 
                                 alpha=0.8, width=0.7)
                axes[0,0].set_title('Delays: Rush Hour vs Weather Conditions', 
                                   fontsize=14, fontweight='bold')
                axes[0,0].set_ylabel('Average Delay (minutes)', fontsize=12)
                axes[0,0].set_xlabel('Time Period', fontsize=12)
                axes[0,0].set_xticklabels(['Regular Hours', 'Rush Hour'], rotation=0)
                axes[0,0].legend(['Clear', 'Rain'], fontsize=11)
                axes[0,0].grid(axis='y', alpha=0.3)
        
        # 2. Hourly patterns with weather overlay
        if 'hour_of_day' in df.columns:
            hourly_all = df.groupby('hour_of_day')['delay_minutes'].mean()
            hourly_rain = df[df['is_raining'] == True].groupby('hour_of_day')['delay_minutes'].mean()
            hourly_clear = df[df['is_raining'] == False].groupby('hour_of_day')['delay_minutes'].mean()
            
            axes[0,1].plot(hourly_all.index, hourly_all.values, 'o-', linewidth=2, 
                          label='All days', color='gray', markersize=4)
            axes[0,1].plot(hourly_rain.index, hourly_rain.values, 's-', linewidth=2, 
                          label='Rainy days', color='blue', markersize=4)
            axes[0,1].plot(hourly_clear.index, hourly_clear.values, '^-', linewidth=2, 
                          label='Clear days', color='orange', markersize=4)
            
            axes[0,1].set_title('Delays by Hour of Day and Weather Conditions', 
                               fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Hour of Day', fontsize=12)
            axes[0,1].set_ylabel('Average Delay (minutes)', fontsize=12)
            axes[0,1].legend(fontsize=11)
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_xticks(range(0, 24, 2))
        
        # 3. Daily weather impact over time
        if 'timestamp' in df.columns:
            df['date'] = df['timestamp'].dt.date
            daily_weather = df.groupby(['date', 'is_raining'])['delay_minutes'].mean().unstack(fill_value=0)
            
            if not daily_weather.empty and len(daily_weather) > 1:
                daily_weather.index = pd.to_datetime(daily_weather.index)
                
                if False in daily_weather.columns:
                    axes[1,0].plot(daily_weather.index, daily_weather[False], 
                                  'o-', alpha=0.7, label='Clear Days', color='orange', markersize=3)
                if True in daily_weather.columns:
                    axes[1,0].plot(daily_weather.index, daily_weather[True], 
                                  's-', alpha=0.7, label='Rainy Days', color='blue', markersize=3)
                
                axes[1,0].set_title('Temporal Evolution: Delays and Weather Conditions', 
                                   fontsize=14, fontweight='bold')
                axes[1,0].set_xlabel('Date', fontsize=12)
                axes[1,0].set_ylabel('Average Delay (minutes)', fontsize=12)
                axes[1,0].legend(fontsize=11)
                axes[1,0].grid(True, alpha=0.3)
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Weekend vs weekday with weather
        if 'is_weekend' in df.columns and 'is_raining' in df.columns:
            weekend_weather = df.groupby(['is_weekend', 'is_raining'])['delay_minutes'].mean().unstack()
            
            if not weekend_weather.empty:
                weekend_weather.plot(kind='bar', ax=axes[1,1], color=['lightgreen', 'darkgreen'], 
                                    alpha=0.8, width=0.7)
                axes[1,1].set_title('Delays: Weekend vs Weekdays with Weather Conditions', 
                                   fontsize=14, fontweight='bold')
                axes[1,1].set_ylabel('Average Delay (minutes)', fontsize=12)
                axes[1,1].set_xlabel('Day Type', fontsize=12)
                axes[1,1].set_xticklabels(['Weekdays', 'Weekend'], rotation=0)
                axes[1,1].legend(['Clear', 'Rain'], fontsize=11)
                axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/02_temporal_weather_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_station_performance_analysis(self):
        """Analyze station performance with weather impact"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Top stations by delay with station names
        if 'delay_minutes' in df.columns and 'station_name' in df.columns:
            station_stats = df.groupby(['station_code', 'station_name']).agg({
                'delay_minutes': ['mean', 'count']
            }).reset_index()
            
            station_stats.columns = ['station_code', 'station_name', 'avg_delay', 'total_records']
            station_stats = station_stats[station_stats['total_records'] >= 200]  # Significant data
            top_delay_stations = station_stats.nlargest(10, 'avg_delay')
            
            # Truncate long station names
            station_names = [name[:20] + '...' if len(name) > 20 else name 
                           for name in top_delay_stations['station_name']]
            
            bars = axes[0,0].barh(range(len(top_delay_stations)), top_delay_stations['avg_delay'],
                                 color='coral', alpha=0.8, edgecolor='black')
            axes[0,0].set_yticks(range(len(top_delay_stations)))
            axes[0,0].set_yticklabels(station_names, fontsize=10)
            axes[0,0].set_title('Top 10 Stations by Average Delay', fontsize=14, fontweight='bold')
            axes[0,0].set_xlabel('Average Delay (minutes)', fontsize=12)
            axes[0,0].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, delay) in enumerate(zip(bars, top_delay_stations['avg_delay'])):
                axes[0,0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                              f'{delay:.1f}', va='center', fontsize=9)
        
        # 2. Station volume analysis
        if 'station_name' in df.columns:
            volume_stats = df.groupby(['station_code', 'station_name']).size().reset_index(name='volume')
            top_volume_stations = volume_stats.nlargest(10, 'volume')
            
            volume_names = [name[:20] + '...' if len(name) > 20 else name 
                           for name in top_volume_stations['station_name']]
            
            axes[0,1].bar(range(len(top_volume_stations)), top_volume_stations['volume'],
                         color='lightgreen', alpha=0.8, edgecolor='black')
            axes[0,1].set_xticks(range(len(top_volume_stations)))
            axes[0,1].set_xticklabels(volume_names, rotation=45, ha='right', fontsize=10)
            axes[0,1].set_title('Top 10 Stations by Traffic Volume', fontsize=14, fontweight='bold')
            axes[0,1].set_ylabel('Number of Trains', fontsize=12)
            axes[0,1].grid(axis='y', alpha=0.3)
        
        # 3. Weather impact by major stations
        if 'station_name' in df.columns and 'is_raining' in df.columns:
            # Get top 8 stations by volume for weather analysis
            major_stations = df.groupby(['station_code', 'station_name']).size().nlargest(8)
            major_station_codes = major_stations.index.get_level_values(0).tolist()
            
            weather_by_station = df[df['station_code'].isin(major_station_codes)].groupby(
                ['station_name', 'is_raining'])['delay_minutes'].mean().unstack(fill_value=0)
            
            if not weather_by_station.empty:
                weather_by_station.plot(kind='bar', ax=axes[1,0], 
                                       color=['lightblue', 'darkblue'], alpha=0.8, width=0.8)
                axes[1,0].set_title('Weather Impact at Major Stations', 
                                   fontsize=14, fontweight='bold')
                axes[1,0].set_ylabel('Average Delay (minutes)', fontsize=12)
                axes[1,0].set_xlabel('Station', fontsize=12)
                axes[1,0].legend(['Clear', 'Rain'], fontsize=11)
                axes[1,0].tick_params(axis='x', rotation=45)
                axes[1,0].grid(axis='y', alpha=0.3)
        
        # 4. Train category performance in bad weather
        if 'train_category' in df.columns and 'is_raining' in df.columns:
            category_weather = df.groupby(['train_category', 'is_raining'])['delay_minutes'].agg(['mean', 'count'])
            category_weather = category_weather[category_weather['count'] >= 100]  # Significant data
            
            if not category_weather.empty:
                category_means = category_weather['mean'].unstack(fill_value=0)
                category_means.plot(kind='bar', ax=axes[1,1], 
                                   color=['lightcoral', 'darkred'], alpha=0.8, width=0.8)
                axes[1,1].set_title('Performance by Train Category and Weather Conditions', 
                                   fontsize=14, fontweight='bold')
                axes[1,1].set_ylabel('Average Delay (minutes)', fontsize=12)
                axes[1,1].set_xlabel('Train Category', fontsize=12)
                axes[1,1].legend(['Clear', 'Rain'], fontsize=11)
                axes[1,1].tick_params(axis='x', rotation=0)
                axes[1,1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/03_station_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self):
        """Generate focused correlation analysis"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        complete_data = df.dropna(subset=['delay_minutes', 'temperature', 'precip_mm', 'wind_speed'])
        
        if complete_data.empty:
            print("No complete data for correlation analysis")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Correlation matrix for key weather variables
        weather_vars = ['delay_minutes', 'temperature', 'precip_mm', 'wind_speed']
        correlation_matrix = complete_data[weather_vars].corr()
        
        # Create custom labels in English
        labels = ['Delays\n(min)', 'Temperature\n(°C)', 'Precipitation\n(mm)', 'Wind\n(km/h)']
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'},
                   xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Correlation Matrix: Weather Variables and Delays', 
                         fontsize=14, fontweight='bold')
        
        # 2. Scatter plot: Precipitation vs Delays (most important relationship)
        # Add some jitter to better show the distribution
        precip_jitter = complete_data['precip_mm'] + np.random.normal(0, 0.01, len(complete_data))
        delay_jitter = complete_data['delay_minutes'] + np.random.normal(0, 0.1, len(complete_data))
        
        # Color points by rain intensity
        colors = complete_data['precip_mm'].apply(lambda x: 'red' if x > 1 else 'orange' if x > 0.1 else 'lightblue')
        
        scatter = axes[1].scatter(precip_jitter, delay_jitter, c=colors, alpha=0.6, s=15)
        
        # Add trend line
        if len(complete_data) > 1:
            z = np.polyfit(complete_data['precip_mm'], complete_data['delay_minutes'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(0, complete_data['precip_mm'].max(), 100)
            axes[1].plot(x_trend, p(x_trend), "r-", linewidth=2, alpha=0.8)
        
        axes[1].set_xlabel('Precipitation (mm)', fontsize=12)
        axes[1].set_ylabel('Delay (minutes)', fontsize=12)
        axes[1].set_title('Precipitation-Delay Relationship', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr_coef = complete_data['precip_mm'].corr(complete_data['delay_minutes'])
        axes[1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                    transform=axes[1].transAxes, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend for colors
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='lightblue', label='None (0 mm)'),
                          Patch(facecolor='orange', label='Light (0.1-1 mm)'),
                          Patch(facecolor='red', label='Heavy (>1 mm)')]
        axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/04_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key correlations
        print("\n" + "="*50)
        print("KEY CORRELATIONS WITH DELAYS:")
        print("="*50)
        for var in ['temperature', 'precip_mm', 'wind_speed']:
            corr = complete_data[var].corr(complete_data['delay_minutes'])
            significance = "***" if abs(corr) > 0.1 else "**" if abs(corr) > 0.05 else "*" if abs(corr) > 0.01 else ""
            var_name = {'temperature': 'Temperature', 'precip_mm': 'Precipitation', 'wind_speed': 'Wind Speed'}[var]
            print(f"  {var_name}: {corr:.4f} {significance}")
        print("\nSignificance levels: *** |r| > 0.1, ** |r| > 0.05, * |r| > 0.01")
    

    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS & KEY FINDINGS")
        print("="*60)
        
        if self.data['integrated'].empty:
            print("No integrated data available for summary statistics")
            return
        
        df = self.data['integrated']
        
        # Basic dataset statistics
        print(f"Total Records: {len(df):,}")
        print(f"Date Range: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"Unique Stations: {df['station_code'].nunique()}")
        print(f"Unique Trains: {df['train_id'].nunique()}")
        
        # Delay statistics
        if 'delay_minutes' in df.columns:
            delay_data = df['delay_minutes'].dropna()
            print(f"\nDelay Statistics:")
            print(f"  Average Delay: {delay_data.mean():.2f} minutes")
            print(f"  Median Delay: {delay_data.median():.2f} minutes")
            print(f"  Max Delay: {delay_data.max():.2f} minutes")
            print(f"  Delayed Trains (>5 min): {(delay_data > 5).sum():,} ({(delay_data > 5).mean()*100:.1f}%)")
        
        # Weather statistics
        weather_cols = ['temperature', 'wind_speed', 'precip_mm']
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if available_weather:
            print(f"\nWeather Statistics:")
            for col in available_weather:
                data = df[col].dropna()
                if not data.empty:
                    print(f"  {col.title()}: Mean={data.mean():.2f}, Range=[{data.min():.2f}, {data.max():.2f}]")
        
        # Data quality summary
        total_possible = len(df) * len(df.columns)
        total_missing = df.isnull().sum().sum()
        completeness = (1 - total_missing / total_possible) * 100
        print(f"\nData Quality:")
        print(f"  Overall Completeness: {completeness:.1f}%")
        print(f"  Weather Match Rate: {(df['temperature'].notna()).mean()*100:.1f}%")
        
        # Temporal patterns
        if 'is_rush_hour' in df.columns and 'delay_minutes' in df.columns:
            rush_delays = df[df['is_rush_hour'] == True]['delay_minutes'].mean()
            non_rush_delays = df[df['is_rush_hour'] == False]['delay_minutes'].mean()
            print(f"\nTemporal Patterns:")
            print(f"  Rush Hour Avg Delay: {rush_delays:.2f} minutes")
            print(f"  Non-Rush Hour Avg Delay: {non_rush_delays:.2f} minutes")
        
        # Weather impact
        if 'is_raining' in df.columns and 'delay_minutes' in df.columns:
            rain_delays = df[df['is_raining'] == True]['delay_minutes'].mean()
            no_rain_delays = df[df['is_raining'] == False]['delay_minutes'].mean()
            print(f"\nWeather Impact:")
            print(f"  Rainy Conditions Avg Delay: {rain_delays:.2f} minutes")
            print(f"  Clear Conditions Avg Delay: {no_rain_delays:.2f} minutes")
    
    def run_complete_eda(self):
        """Run the focused EDA analysis"""
        print("EDA Analysis: Weather-Railway Delay Correlation in Lombardy")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Generate focused analyses
        self.data_overview()
        
        print("\nGenerating weather impact analysis...")
        self.plot_weather_impact_analysis()
        
        print("\nGenerating temporal pattern analysis...")
        self.plot_temporal_weather_patterns()
        
        print("\nGenerating station performance analysis...")
        self.plot_station_performance_analysis()
        
        print("\nGenerating correlation analysis...")
        self.plot_correlation_analysis()
        
        self.generate_summary_statistics()
        
        print(f"\nEDA COMPLETED!")
        print(f"Figures saved in '{self.figures_dir}/'")
        print("\nRESEARCH QUESTIONS ADDRESSED:")
        print("1. Correlation between weather conditions and railway delays")
        print("2. Temporal patterns (rush hours, days of the week)")
        print("3. Spatial analysis (station performance)")
        print("4. Operational impact (train categories, combined conditions)")
        print("5. Quantification of bad weather effect on delays")


if __name__ == "__main__":
    # Run the complete EDA
    eda = RailwayWeatherEDA()
    eda.run_complete_eda()