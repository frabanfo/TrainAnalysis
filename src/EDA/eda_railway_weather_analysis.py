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
    """Comprehensive EDA for Railway-Weather Correlation Analysis"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.data = {}
        self.figures_dir = "eda_figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        
    def load_data(self):
        """Load all necessary data from database"""
        print("Loading data from database...")
        
        # Load integrated data (main dataset)
        integrated_query = """
        SELECT 
            train_id, timestamp, station_code, delay_minutes, temperature,
            wind_speed, precip_mm, weather_code, train_category, route,
            delay_status, destination, is_cancelled, hour_of_day, day_of_week,
            is_weekend, is_rush_hour, temp_category, is_raining, rain_intensity,
            wind_category, is_delayed, delay_category
        FROM train_weather_integrated
        ORDER BY timestamp
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
        
        # Missing values analysis
        print("\nMissing Values Analysis:")
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing_pct.index,
            'Missing_Count': df.isnull().sum().values,
            'Missing_Percentage': missing_pct.values
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(missing_df.to_string(index=False))
        
        # Basic statistics for key variables
        print("\nKey Variable Statistics:")
        numeric_cols = ['delay_minutes', 'temperature', 'wind_speed', 'precip_mm']
        available_cols = [col for col in numeric_cols if col in df.columns]
        if available_cols:
            print(df[available_cols].describe().round(2))
    
    def plot_missing_values(self):
        """Plot missing values analysis"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Missing values bar chart
        missing_pct = (df.isnull().sum() / len(df) * 100)
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=True)
        
        if not missing_pct.empty:
            missing_pct.plot(kind='barh', ax=ax1, color='coral')
            ax1.set_title('Missing Values by Column (%)')
            ax1.set_xlabel('Percentage Missing')
            
            # Missing values heatmap
            missing_matrix = df.isnull()
            if missing_matrix.any().any():
                sns.heatmap(missing_matrix.iloc[:1000], ax=ax2, cbar=True, 
                           yticklabels=False, cmap='viridis')
                ax2.set_title('Missing Values Pattern (First 1000 records)')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/01_missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_delay_distribution(self):
        """Analyze delay distribution patterns"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        if 'delay_minutes' not in df.columns or df['delay_minutes'].isnull().all():
            print("No delay data available for analysis")
            return
        
        # Remove extreme outliers for better visualization
        delay_data = df['delay_minutes'].dropna()
        q99 = delay_data.quantile(0.99)
        delay_data_clean = delay_data[delay_data <= q99]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histogram of delays
        axes[0,0].hist(delay_data_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Distribution of Train Delays')
        axes[0,0].set_xlabel('Delay (minutes)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(delay_data_clean.mean(), color='red', linestyle='--', 
                         label=f'Mean: {delay_data_clean.mean():.1f} min')
        axes[0,0].legend()
        
        # Box plot by delay status
        if 'delay_status' in df.columns:
            delay_status_data = df[df['delay_minutes'].notna() & df['delay_status'].notna()]
            if not delay_status_data.empty:
                sns.boxplot(data=delay_status_data, x='delay_status', y='delay_minutes', ax=axes[0,1])
                axes[0,1].set_title('Delays by Status Category')
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # Delays by hour of day
        if 'hour_of_day' in df.columns:
            hourly_delays = df.groupby('hour_of_day')['delay_minutes'].agg(['mean', 'count']).reset_index()
            axes[1,0].bar(hourly_delays['hour_of_day'], hourly_delays['mean'], 
                         alpha=0.7, color='lightgreen')
            axes[1,0].set_title('Average Delays by Hour of Day')
            axes[1,0].set_xlabel('Hour of Day')
            axes[1,0].set_ylabel('Average Delay (minutes)')
        
        # Delays by day of week
        if 'day_of_week' in df.columns:
            daily_delays = df.groupby('day_of_week')['delay_minutes'].agg(['mean', 'count']).reset_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[1,1].bar(range(len(daily_delays)), daily_delays['mean'], 
                         alpha=0.7, color='orange')
            axes[1,1].set_title('Average Delays by Day of Week')
            axes[1,1].set_xlabel('Day of Week')
            axes[1,1].set_ylabel('Average Delay (minutes)')
            axes[1,1].set_xticks(range(len(day_names)))
            axes[1,1].set_xticklabels(day_names)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/02_delay_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weather_delay_correlation(self):
        """Analyze correlation between weather conditions and delays"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        # Filter data with both delay and weather information
        complete_data = df.dropna(subset=['delay_minutes', 'temperature'])
        
        if complete_data.empty:
            print("No complete weather-delay data available for correlation analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature vs Delay scatter plot
        if 'temperature' in complete_data.columns:
            axes[0,0].scatter(complete_data['temperature'], complete_data['delay_minutes'], 
                             alpha=0.5, s=20)
            axes[0,0].set_xlabel('Temperature (°C)')
            axes[0,0].set_ylabel('Delay (minutes)')
            axes[0,0].set_title('Temperature vs Train Delays')
            
            # Add trend line
            z = np.polyfit(complete_data['temperature'], complete_data['delay_minutes'], 1)
            p = np.poly1d(z)
            axes[0,0].plot(complete_data['temperature'], p(complete_data['temperature']), 
                          "r--", alpha=0.8)
        
        # Precipitation vs Delay scatter plot
        if 'precip_mm' in complete_data.columns:
            precip_data = complete_data[complete_data['precip_mm'].notna()]
            if not precip_data.empty:
                axes[0,1].scatter(precip_data['precip_mm'], precip_data['delay_minutes'], 
                                 alpha=0.5, s=20, color='blue')
                axes[0,1].set_xlabel('Precipitation (mm)')
                axes[0,1].set_ylabel('Delay (minutes)')
                axes[0,1].set_title('Precipitation vs Train Delays')
        
        # Wind speed vs Delay scatter plot
        if 'wind_speed' in complete_data.columns:
            wind_data = complete_data[complete_data['wind_speed'].notna()]
            if not wind_data.empty:
                axes[1,0].scatter(wind_data['wind_speed'], wind_data['delay_minutes'], 
                                 alpha=0.5, s=20, color='green')
                axes[1,0].set_xlabel('Wind Speed (km/h)')
                axes[1,0].set_ylabel('Delay (minutes)')
                axes[1,0].set_title('Wind Speed vs Train Delays')
        
        # Weather condition categories vs delays
        if 'is_raining' in complete_data.columns:
            rain_delays = complete_data.groupby('is_raining')['delay_minutes'].agg(['mean', 'count'])
            rain_labels = ['No Rain', 'Rain']
            axes[1,1].bar(range(len(rain_delays)), rain_delays['mean'], 
                         color=['lightblue', 'darkblue'], alpha=0.7)
            axes[1,1].set_title('Average Delays: Rain vs No Rain')
            axes[1,1].set_ylabel('Average Delay (minutes)')
            axes[1,1].set_xticks(range(len(rain_labels)))
            axes[1,1].set_xticklabels(rain_labels)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/03_weather_delay_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_temporal_patterns(self):
        """Analyze temporal patterns in delays and weather"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Daily delay patterns
        if 'timestamp' in df.columns and 'delay_minutes' in df.columns:
            df['date'] = df['timestamp'].dt.date
            daily_stats = df.groupby('date').agg({
                'delay_minutes': ['mean', 'count'],
                'temperature': 'mean',
                'precip_mm': 'sum'
            }).reset_index()
            
            daily_stats.columns = ['date', 'avg_delay', 'train_count', 'avg_temp', 'total_precip']
            daily_stats['date'] = pd.to_datetime(daily_stats['date'])
            
            # Plot daily average delays
            axes[0,0].plot(daily_stats['date'], daily_stats['avg_delay'], 
                          marker='o', markersize=3, linewidth=1)
            axes[0,0].set_title('Daily Average Delays Over Time')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Average Delay (minutes)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Rush hour analysis
        if 'is_rush_hour' in df.columns and 'delay_minutes' in df.columns:
            rush_hour_stats = df.groupby('is_rush_hour')['delay_minutes'].agg(['mean', 'count'])
            rush_labels = ['Non-Rush Hour', 'Rush Hour']
            axes[0,1].bar(range(len(rush_hour_stats)), rush_hour_stats['mean'], 
                         color=['lightcoral', 'darkred'], alpha=0.7)
            axes[0,1].set_title('Average Delays: Rush Hour vs Non-Rush Hour')
            axes[0,1].set_ylabel('Average Delay (minutes)')
            axes[0,1].set_xticks(range(len(rush_labels)))
            axes[0,1].set_xticklabels(rush_labels)
        
        # Weekend vs weekday analysis
        if 'is_weekend' in df.columns and 'delay_minutes' in df.columns:
            weekend_stats = df.groupby('is_weekend')['delay_minutes'].agg(['mean', 'count'])
            weekend_labels = ['Weekday', 'Weekend']
            axes[1,0].bar(range(len(weekend_stats)), weekend_stats['mean'], 
                         color=['lightgreen', 'darkgreen'], alpha=0.7)
            axes[1,0].set_title('Average Delays: Weekday vs Weekend')
            axes[1,0].set_ylabel('Average Delay (minutes)')
            axes[1,0].set_xticks(range(len(weekend_labels)))
            axes[1,0].set_xticklabels(weekend_labels)
        
        # Hourly heatmap
        if 'hour_of_day' in df.columns and 'day_of_week' in df.columns:
            hourly_heatmap = df.pivot_table(
                values='delay_minutes', 
                index='day_of_week', 
                columns='hour_of_day', 
                aggfunc='mean'
            )
            
            if not hourly_heatmap.empty:
                sns.heatmap(hourly_heatmap, ax=axes[1,1], cmap='YlOrRd', 
                           cbar_kws={'label': 'Average Delay (min)'})
                axes[1,1].set_title('Average Delays Heatmap (Day vs Hour)')
                axes[1,1].set_xlabel('Hour of Day')
                axes[1,1].set_ylabel('Day of Week')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/04_temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_station_analysis(self):
        """Analyze delays by station and geographic patterns"""
        if self.data['integrated'].empty or self.data['stations'].empty:
            return
        
        df = self.data['integrated']
        stations_df = self.data['stations']
        
        # Merge with station information
        df_with_stations = df.merge(stations_df, on='station_code', how='left')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Top stations by average delay
        if 'delay_minutes' in df.columns:
            station_delays = df.groupby('station_code').agg({
                'delay_minutes': ['mean', 'count'],
                'train_id': 'nunique'
            }).reset_index()
            
            station_delays.columns = ['station_code', 'avg_delay', 'total_trains', 'unique_trains']
            station_delays = station_delays[station_delays['total_trains'] >= 10]  # Filter for significance
            top_stations = station_delays.nlargest(15, 'avg_delay')
            
            axes[0,0].barh(range(len(top_stations)), top_stations['avg_delay'])
            axes[0,0].set_yticks(range(len(top_stations)))
            axes[0,0].set_yticklabels(top_stations['station_code'])
            axes[0,0].set_title('Top 15 Stations by Average Delay')
            axes[0,0].set_xlabel('Average Delay (minutes)')
        
        # Station delay distribution
        if 'delay_minutes' in df.columns:
            station_delay_counts = df.groupby('station_code')['delay_minutes'].count().sort_values(ascending=False)
            top_volume_stations = station_delay_counts.head(10)
            
            axes[0,1].bar(range(len(top_volume_stations)), top_volume_stations.values)
            axes[0,1].set_xticks(range(len(top_volume_stations)))
            axes[0,1].set_xticklabels(top_volume_stations.index, rotation=45)
            axes[0,1].set_title('Top 10 Stations by Train Volume')
            axes[0,1].set_ylabel('Number of Train Records')
        
        # Train category analysis
        if 'train_category' in df.columns and 'delay_minutes' in df.columns:
            category_delays = df.groupby('train_category')['delay_minutes'].agg(['mean', 'count'])
            category_delays = category_delays[category_delays['count'] >= 10]
            
            if not category_delays.empty:
                axes[1,0].bar(range(len(category_delays)), category_delays['mean'])
                axes[1,0].set_xticks(range(len(category_delays)))
                axes[1,0].set_xticklabels(category_delays.index, rotation=45)
                axes[1,0].set_title('Average Delays by Train Category')
                axes[1,0].set_ylabel('Average Delay (minutes)')
        
        # Delay status distribution
        if 'delay_status' in df.columns:
            status_counts = df['delay_status'].value_counts()
            axes[1,1].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
            axes[1,1].set_title('Distribution of Delay Status')
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/05_station_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_data_quality_trends(self):
        """Analyze data quality metrics and ingestion trends"""
        if self.data['dq_metrics'].empty:
            print("No data quality metrics available")
            return
        
        dq_df = self.data['dq_metrics']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Data ingestion volume over time
        if not self.data['trains'].empty:
            trains_df = self.data['trains'].copy()
            trains_df['date'] = trains_df['timestamp'].dt.date
            daily_ingestion = trains_df.groupby('date').size().reset_index(name='count')
            daily_ingestion['date'] = pd.to_datetime(daily_ingestion['date'])
            
            axes[0,0].plot(daily_ingestion['date'], daily_ingestion['count'], 
                          marker='o', markersize=3)
            axes[0,0].set_title('Daily Train Data Ingestion Volume')
            axes[0,0].set_xlabel('Date')
            axes[0,0].set_ylabel('Number of Records')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # Data quality metrics over time
        if 'metric_name' in dq_df.columns:
            quality_metrics = dq_df.pivot_table(
                values='metric_value', 
                index='timestamp', 
                columns='metric_name', 
                aggfunc='mean'
            )
            
            if not quality_metrics.empty:
                for col in quality_metrics.columns[:3]:  # Plot first 3 metrics
                    axes[0,1].plot(quality_metrics.index, quality_metrics[col], 
                                  marker='o', markersize=2, label=col)
                axes[0,1].set_title('Data Quality Metrics Trends')
                axes[0,1].set_xlabel('Date')
                axes[0,1].set_ylabel('Metric Value')
                axes[0,1].legend()
                axes[0,1].tick_params(axis='x', rotation=45)
        
        # Missing data patterns
        if not self.data['integrated'].empty:
            df = self.data['integrated']
            missing_by_date = df.groupby(df['timestamp'].dt.date).apply(
                lambda x: x.isnull().sum() / len(x) * 100
            ).reset_index()
            
            if len(missing_by_date) > 1:
                key_cols = ['temperature', 'wind_speed', 'precip_mm']
                available_cols = [col for col in key_cols if col in missing_by_date.columns]
                
                for col in available_cols[:2]:  # Plot first 2 available columns
                    axes[1,0].plot(missing_by_date['timestamp'], missing_by_date[col], 
                                  marker='o', markersize=2, label=f'{col} missing %')
                axes[1,0].set_title('Missing Data Trends Over Time')
                axes[1,0].set_xlabel('Date')
                axes[1,0].set_ylabel('Missing Percentage')
                axes[1,0].legend()
                axes[1,0].tick_params(axis='x', rotation=45)
        
        # Data completeness by table
        completeness_data = []
        for table_name, df in self.data.items():
            if table_name in ['integrated', 'trains', 'weather'] and not df.empty:
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                completeness_data.append({'table': table_name, 'completeness': completeness})
        
        if completeness_data:
            comp_df = pd.DataFrame(completeness_data)
            axes[1,1].bar(comp_df['table'], comp_df['completeness'], 
                         color=['skyblue', 'lightgreen', 'coral'])
            axes[1,1].set_title('Data Completeness by Table')
            axes[1,1].set_ylabel('Completeness (%)')
            axes[1,1].set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/06_data_quality_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_correlation_matrix(self):
        """Generate correlation matrix for key variables"""
        if self.data['integrated'].empty:
            return
        
        df = self.data['integrated']
        
        # Select numeric columns for correlation analysis
        numeric_cols = ['delay_minutes', 'temperature', 'wind_speed', 'precip_mm', 
                       'hour_of_day', 'day_of_week']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print("Insufficient numeric data for correlation analysis")
            return
        
        correlation_data = df[available_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0, 
                   square=True, fmt='.3f')
        plt.title('Correlation Matrix: Weather and Delay Variables')
        plt.tight_layout()
        plt.savefig(f'{self.figures_dir}/07_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print key correlations
        print("\nKey Correlations with Delay Minutes:")
        delay_corr = correlation_data['delay_minutes'].drop('delay_minutes').sort_values(key=abs, ascending=False)
        for var, corr in delay_corr.items():
            print(f"  {var}: {corr:.3f}")
    
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
        """Run the complete EDA analysis"""
        print("Starting Comprehensive Railway-Weather EDA Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Generate all analyses
        self.data_overview()
        self.plot_missing_values()
        self.plot_delay_distribution()
        self.plot_weather_delay_correlation()
        self.plot_temporal_patterns()
        self.plot_station_analysis()
        self.plot_data_quality_trends()
        self.generate_correlation_matrix()
        self.generate_summary_statistics()
        
        print(f"\nEDA Complete! All figures saved to '{self.figures_dir}/' directory")
        print("\nKey Research Questions Addressed:")
        print("1. ✓ Correlation between meteorological conditions and railway delays")
        print("2. ✓ Temporal patterns (hourly, daily, seasonal)")
        print("3. ✓ Spatial patterns (station-level analysis)")
        print("4. ✓ Operational aspects (train categories, rush hours)")
        print("5. ✓ Data quality and integration success metrics")


if __name__ == "__main__":
    # Run the complete EDA
    eda = RailwayWeatherEDA()
    eda.run_complete_eda()