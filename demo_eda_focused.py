#!/usr/bin/env python3
"""
Demo EDA Focalizzato - Correlazione Tempo-Ritardi Ferroviari
============================================================

Versione demo migliorata che enfatizza la relazione tra maltempo e ritardi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def generate_realistic_weather_delay_data():
    """Generate realistic sample data emphasizing weather-delay correlation"""
    print("Generazione dati demo realistici per analisi tempo-ritardi...")
    
    # Date range for the last 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Generate timestamps (every 2 hours for more realistic data)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='2H')
    
    # Sample station codes and names (Lombardy region)
    stations_data = [
        ('MI01', 'Milano Centrale'), ('MI02', 'Milano Porta Garibaldi'), 
        ('BG01', 'Bergamo'), ('BS01', 'Brescia'), ('CO01', 'Como San Giovanni'),
        ('CR01', 'Cremona'), ('LC01', 'Lecco'), ('LO01', 'Lodi'),
        ('MN01', 'Mantova'), ('PV01', 'Pavia'), ('SO01', 'Sondrio'), ('VA01', 'Varese')
    ]
    
    # Generate sample data with stronger weather correlation
    np.random.seed(42)  # For reproducible results
    
    data = []
    for timestamp in timestamps:
        for station_code, station_name in stations_data:
            # Skip some records to simulate missing data
            if np.random.random() < 0.1:  # 10% missing data
                continue
                
            # Generate weather data with seasonal patterns
            base_temp = 8 + 8 * np.sin(2 * np.pi * timestamp.dayofyear / 365)  # Seasonal variation
            temperature = base_temp + np.random.normal(0, 4)
            
            # Rain probability varies by season and creates more realistic patterns
            season_rain_prob = 0.4 if timestamp.month in [11, 12, 1, 2] else 0.25
            is_raining = np.random.random() < season_rain_prob
            
            # More realistic precipitation amounts
            if is_raining:
                precip_mm = np.random.lognormal(0.5, 0.8)  # Log-normal for realistic rain amounts
                precip_mm = min(precip_mm, 15)  # Cap at 15mm
            else:
                precip_mm = 0
            
            wind_speed = np.random.gamma(2, 2.5)  # Realistic wind speed distribution
            weather_code = 61 if is_raining else np.random.choice([0, 1, 2, 3], p=[0.5, 0.2, 0.2, 0.1])
            
            # Generate train data with STRONGER weather correlation
            num_trains = np.random.poisson(2)  # Average 2 trains per 2-hour period per station
            
            for train_idx in range(num_trains):
                train_id = f"T{station_code}_{timestamp.strftime('%Y%m%d_%H')}_{train_idx:02d}"
                
                # Base delay influenced more strongly by weather
                base_delay = 1.5  # Base 1.5-minute delay
                
                # STRONGER weather impact for demo
                if is_raining:
                    if precip_mm > 5:  # Heavy rain
                        base_delay += np.random.exponential(8)  # Heavy rain causes significant delays
                    elif precip_mm > 1:  # Moderate rain
                        base_delay += np.random.exponential(4)  # Moderate impact
                    else:  # Light rain
                        base_delay += np.random.exponential(2)  # Light impact
                
                if wind_speed > 12:
                    base_delay += np.random.exponential(3)  # High wind increases delays
                
                if temperature < 0:
                    base_delay += np.random.exponential(4)  # Freezing conditions
                elif temperature > 30:
                    base_delay += np.random.exponential(2)  # Very hot conditions
                
                # Time-based factors
                hour = timestamp.hour
                is_rush_hour = hour in [7, 8, 9, 17, 18, 19]
                is_weekend = timestamp.weekday() >= 5
                
                if is_rush_hour and not is_weekend:
                    base_delay += np.random.exponential(3)  # Rush hour delays
                    
                    # Combined effect: rain + rush hour = much worse
                    if is_raining:
                        base_delay += np.random.exponential(2)  # Compound effect
                
                # Add random variation
                delay_minutes = max(0, base_delay + np.random.normal(0, 1.5))
                
                # Determine delay status
                if delay_minutes <= 2:
                    delay_status = 'on_time'
                elif delay_minutes <= 5:
                    delay_status = 'delayed'
                else:
                    delay_status = 'delayed'
                
                # Random cancellations (more likely in bad weather)
                cancel_prob = 0.05 if is_raining and precip_mm > 3 else 0.01
                is_cancelled = np.random.random() < cancel_prob
                if is_cancelled:
                    delay_status = 'cancelled'
                    delay_minutes = np.nan
                
                # Train categories
                train_category = np.random.choice(['REG', 'IC', 'FR', 'RV'], 
                                                p=[0.5, 0.2, 0.15, 0.15])
                
                record = {
                    'train_id': train_id,
                    'timestamp': timestamp,
                    'station_code': station_code,
                    'station_name': station_name,
                    'delay_minutes': delay_minutes,
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'precip_mm': precip_mm,
                    'weather_code': weather_code,
                    'train_category': train_category,
                    'delay_status': delay_status,
                    'is_cancelled': is_cancelled,
                    'hour_of_day': hour,
                    'day_of_week': timestamp.weekday(),
                    'is_weekend': is_weekend,
                    'is_rush_hour': is_rush_hour,
                    'is_raining': is_raining,
                    'is_delayed': delay_minutes > 5 if not pd.isna(delay_minutes) else False,
                }
                
                data.append(record)
    
    df = pd.DataFrame(data)
    print(f"Generati {len(df):,} record demo")
    print(f"Periodo: {df['timestamp'].min()} - {df['timestamp'].max()}")
    print(f"Stazioni: {df['station_code'].nunique()}")
    
    return df

def run_focused_weather_analysis():
    """Run focused weather-delay analysis"""
    print("\n" + "="*70)
    print("ANALISI EDA FOCALIZZATA: IMPATTO METEO SUI RITARDI FERROVIARI")
    print("="*70)
    
    # Generate sample data
    df = generate_realistic_weather_delay_data()
    
    # Create figures directory
    import os
    figures_dir = "demo_focused_figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Key statistics
    complete_data = df.dropna(subset=['delay_minutes'])
    
    print(f"\n📊 STATISTICHE CHIAVE:")
    print(f"   • Record totali: {len(df):,}")
    print(f"   • Ritardo medio: {complete_data['delay_minutes'].mean():.2f} minuti")
    print(f"   • Treni in ritardo (>5 min): {(complete_data['delay_minutes'] > 5).mean()*100:.1f}%")
    print(f"   • Giorni con pioggia: {df['is_raining'].mean()*100:.1f}%")
    
    # Weather impact analysis
    rain_delays = complete_data[complete_data['is_raining'] == True]['delay_minutes'].mean()
    clear_delays = complete_data[complete_data['is_raining'] == False]['delay_minutes'].mean()
    rain_impact = ((rain_delays - clear_delays) / clear_delays) * 100
    
    temp_corr = df['temperature'].corr(df['delay_minutes'])
    precip_corr = df['precip_mm'].corr(df['delay_minutes'])
    wind_corr = df['wind_speed'].corr(df['delay_minutes'])
    
    print(f"\n🌧️ IMPATTO METEOROLOGICO:")
    print(f"   • Ritardi con pioggia: {rain_delays:.2f} min")
    print(f"   • Ritardi con sereno: {clear_delays:.2f} min")
    print(f"   • Aumento ritardi con pioggia: +{rain_impact:.1f}%")
    print(f"   • Correlazione precipitazioni-ritardi: {precip_corr:.3f}")
    
    # Generate focused visualizations
    print(f"\n📈 GENERAZIONE VISUALIZZAZIONI FOCALIZZATE...")
    
    # 1. Main weather impact visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Rain impact comparison
    rain_comparison = complete_data.groupby('is_raining').agg({
        'delay_minutes': ['mean', 'std', 'count']
    }).round(2)
    rain_comparison.columns = ['mean_delay', 'std_delay', 'count']
    
    rain_labels = ['Tempo Sereno', 'Tempo Piovoso']
    colors = ['lightblue', 'darkblue']
    bars = axes[0,0].bar(range(len(rain_comparison)), rain_comparison['mean_delay'], 
                        yerr=rain_comparison['std_delay'], capsize=8,
                        color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    for i, (bar, mean_val, count) in enumerate(zip(bars, rain_comparison['mean_delay'], rain_comparison['count'])):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                      f'{mean_val:.1f} min\\n(n={count:,})', ha='center', va='bottom', fontweight='bold')
    
    axes[0,0].set_title('🌧️ IMPATTO DELLA PIOGGIA SUI RITARDI', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Ritardo Medio (minuti)', fontsize=12)
    axes[0,0].set_xticks(range(len(rain_labels)))
    axes[0,0].set_xticklabels(rain_labels, fontsize=12)
    axes[0,0].grid(axis='y', alpha=0.3)
    
    # Add impact percentage
    axes[0,0].text(0.5, 0.95, f'Aumento: +{rain_impact:.1f}%', 
                  transform=axes[0,0].transAxes, ha='center', va='top',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                  fontsize=12, fontweight='bold')
    
    # Precipitation intensity analysis
    complete_data['precip_category'] = pd.cut(complete_data['precip_mm'], 
                                             bins=[-0.1, 0, 0.5, 2, 10], 
                                             labels=['Assente', 'Leggera', 'Moderata', 'Intensa'])
    
    precip_delays = complete_data.groupby('precip_category')['delay_minutes'].agg(['mean', 'count'])
    precip_delays = precip_delays[precip_delays['count'] >= 10]
    
    if not precip_delays.empty:
        colors_precip = ['lightgreen', 'yellow', 'orange', 'red'][:len(precip_delays)]
        axes[0,1].bar(range(len(precip_delays)), precip_delays['mean'], 
                     color=colors_precip, alpha=0.8, edgecolor='black')
        axes[0,1].set_title('☔ RITARDI PER INTENSITÀ PIOGGIA', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('Ritardo Medio (minuti)', fontsize=12)
        axes[0,1].set_xticks(range(len(precip_delays)))
        axes[0,1].set_xticklabels(precip_delays.index, fontsize=12)
        axes[0,1].grid(axis='y', alpha=0.3)
    
    # Rush hour + weather interaction
    rush_weather = complete_data.groupby(['is_rush_hour', 'is_raining'])['delay_minutes'].mean().unstack()
    if not rush_weather.empty:
        rush_weather.plot(kind='bar', ax=axes[1,0], color=['lightcoral', 'darkred'], 
                         alpha=0.8, width=0.7)
        axes[1,0].set_title('⏰ ORA DI PUNTA + CONDIZIONI METEO', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Ritardo Medio (minuti)', fontsize=12)
        axes[1,0].set_xticklabels(['Ore Normali', 'Ora di Punta'], rotation=0)
        axes[1,0].legend(['Sereno', 'Pioggia'], fontsize=11)
        axes[1,0].grid(axis='y', alpha=0.3)
    
    # Top affected stations
    station_weather_impact = []
    for station in df['station_name'].unique():
        station_data = complete_data[complete_data['station_name'] == station]
        if len(station_data) > 50:  # Sufficient data
            rain_delay = station_data[station_data['is_raining']]['delay_minutes'].mean()
            clear_delay = station_data[~station_data['is_raining']]['delay_minutes'].mean()
            if not pd.isna(rain_delay) and not pd.isna(clear_delay) and clear_delay > 0:
                impact = ((rain_delay - clear_delay) / clear_delay) * 100
                station_weather_impact.append({
                    'station': station[:15] + '...' if len(station) > 15 else station,
                    'impact': impact,
                    'rain_delay': rain_delay
                })
    
    if station_weather_impact:
        impact_df = pd.DataFrame(station_weather_impact).sort_values('impact', ascending=False).head(8)
        bars = axes[1,1].barh(range(len(impact_df)), impact_df['impact'], 
                             color='steelblue', alpha=0.8, edgecolor='black')
        axes[1,1].set_yticks(range(len(impact_df)))
        axes[1,1].set_yticklabels(impact_df['station'], fontsize=10)
        axes[1,1].set_title('🚉 STAZIONI PIÙ COLPITE DAL MALTEMPO', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Aumento % Ritardi con Pioggia', fontsize=12)
        axes[1,1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/weather_impact_focused_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation and scatter analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Precipitation vs delays scatter
    precip_jitter = complete_data['precip_mm'] + np.random.normal(0, 0.02, len(complete_data))
    delay_jitter = complete_data['delay_minutes'] + np.random.normal(0, 0.1, len(complete_data))
    
    colors = complete_data['precip_mm'].apply(lambda x: 'red' if x > 2 else 'orange' if x > 0.5 else 'lightblue')
    
    axes[0].scatter(precip_jitter, delay_jitter, c=colors, alpha=0.6, s=20)
    
    # Add trend line
    if len(complete_data) > 1:
        z = np.polyfit(complete_data['precip_mm'], complete_data['delay_minutes'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, complete_data['precip_mm'].max(), 100)
        axes[0].plot(x_trend, p(x_trend), "r-", linewidth=3, alpha=0.8)
    
    axes[0].set_xlabel('Precipitazioni (mm)', fontsize=12)
    axes[0].set_ylabel('Ritardo (minuti)', fontsize=12)
    axes[0].set_title('🌧️ CORRELAZIONE PIOGGIA-RITARDI', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[0].text(0.05, 0.95, f'Correlazione: {precip_corr:.3f}', 
                transform=axes[0].transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hourly patterns with weather
    hourly_all = complete_data.groupby('hour_of_day')['delay_minutes'].mean()
    hourly_rain = complete_data[complete_data['is_raining']].groupby('hour_of_day')['delay_minutes'].mean()
    hourly_clear = complete_data[~complete_data['is_raining']].groupby('hour_of_day')['delay_minutes'].mean()
    
    axes[1].plot(hourly_all.index, hourly_all.values, 'o-', linewidth=3, 
                label='Media generale', color='gray', markersize=6)
    axes[1].plot(hourly_rain.index, hourly_rain.values, 's-', linewidth=3, 
                label='Giorni piovosi', color='blue', markersize=6)
    axes[1].plot(hourly_clear.index, hourly_clear.values, '^-', linewidth=3, 
                label='Giorni sereni', color='orange', markersize=6)
    
    axes[1].set_title('⏰ PATTERN ORARI CON CONDIZIONI METEO', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Ora del Giorno', fontsize=12)
    axes[1].set_ylabel('Ritardo Medio (minuti)', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(f'{figures_dir}/correlation_temporal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ ANALISI COMPLETATA!")
    print(f"📁 Grafici salvati in '{figures_dir}/'")
    print(f"\n🎯 RISULTATI CHIAVE:")
    print(f"   • La pioggia aumenta i ritardi del {rain_impact:.1f}%")
    print(f"   • Correlazione precipitazioni-ritardi: {precip_corr:.3f}")
    print(f"   • Effetto combinato pioggia + ora di punta particolarmente critico")
    print(f"   • Alcune stazioni sono più vulnerabili alle condizioni meteorologiche")
    
    return df

if __name__ == "__main__":
    # Run the focused weather analysis
    df = run_focused_weather_analysis()