#!/usr/bin/env python3
"""
Quick EDA runner to check data availability and generate basic analysis
"""

import sys
import os
sys.path.append('src')

from database.db_manager import DatabaseManager
import pandas as pd

def check_data_availability():
    """Check what data is available in the database"""
    print("Checking data availability...")
    
    try:
        db_manager = DatabaseManager()
        
        # Check each table
        tables = {
            'stations': 'SELECT COUNT(*) as count FROM stations',
            'trains': 'SELECT COUNT(*) as count FROM trains',
            'weather': 'SELECT COUNT(*) as count FROM weather', 
            'integrated': 'SELECT COUNT(*) as count FROM train_weather_integrated'
        }
        
        for table_name, query in tables.items():
            try:
                result = db_manager.execute_query(query)
                count = result.iloc[0]['count'] if not result.empty else 0
                print(f"  {table_name}: {count:,} records")
            except Exception as e:
                print(f"  {table_name}: Error - {str(e)}")
        
        # Check date ranges if data exists
        try:
            date_query = """
            SELECT 
                MIN(timestamp) as min_date,
                MAX(timestamp) as max_date,
                COUNT(*) as total_records
            FROM train_weather_integrated
            """
            date_result = db_manager.execute_query(date_query)
            if not date_result.empty and date_result.iloc[0]['total_records'] > 0:
                print(f"\nIntegrated data date range:")
                print(f"  From: {date_result.iloc[0]['min_date']}")
                print(f"  To: {date_result.iloc[0]['max_date']}")
                print(f"  Total records: {date_result.iloc[0]['total_records']:,}")
                return True
            else:
                print("\nNo integrated data available for analysis")
                return False
                
        except Exception as e:
            print(f"\nError checking date ranges: {str(e)}")
            return False
            
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

def run_basic_eda():
    """Run basic EDA if data is available"""
    if not check_data_availability():
        print("\nInsufficient data for EDA. Please ensure the data pipeline has run successfully.")
        return
    
    print("\n" + "="*60)
    print("RUNNING BASIC EDA")
    print("="*60)
    
    try:
        # Import and run the EDA class
        from eda_railway_weather_analysis import RailwayWeatherEDA
        
        eda = RailwayWeatherEDA()
        eda.load_data()
        
        if eda.data['integrated'].empty:
            print("No integrated data available. Please run the data pipeline first.")
            return
        
        # Run basic analysis
        eda.data_overview()
        eda.generate_summary_statistics()
        
        print(f"\nFor complete analysis with visualizations, run:")
        print(f"  python eda_railway_weather_analysis.py")
        print(f"Or open the Jupyter notebook:")
        print(f"  jupyter notebook Railway_Weather_EDA.ipynb")
        
    except Exception as e:
        print(f"Error running EDA: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements-eda.txt")

if __name__ == "__main__":
    run_basic_eda()