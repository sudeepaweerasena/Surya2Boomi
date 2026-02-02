#!/usr/bin/env python3
"""
Generate a 24-hour solar flare forecast CSV file
This creates realistic-looking forecast data for testing purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

def generate_24hour_forecast():
    """Generate 24-hour forecast with realistic probabilities"""
    
    # Start time
    base_time = datetime.now(timezone.utc)
    
    # Generate forecast data
    forecast_data = []
    
    for hour in range(24):
        # Create timestamp
        forecast_time = base_time + timedelta(hours=hour)
        
        # Generate realistic probabilities with some variation
        # Base probability with diurnal pattern and random noise
        hour_of_day = (base_time.hour + hour) % 24
        
        # Solar activity peaks during solar noon (local time at sun)
        diurnal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Add some random variation to make it interesting
        random_factor = np.random.uniform(0.1, 2.0)
        
        # Base probability (adjust this to control overall risk level)
        base_prob = np.random.uniform(0.05, 0.35)
        
        # Calculate flare probability
        flare_prob = np.clip(base_prob * diurnal_factor * random_factor, 0.0, 0.95)
        no_flare_prob = 1.0 - flare_prob
        
        # Determine flare class based on probability
        if flare_prob > 0.7:
            flare_class = 'M'  # M-class flare (moderate)
        elif flare_prob > 0.4:
            flare_class = 'C'  # C-class flare (common)
        else:
            flare_class = 'B'  # B-class flare (background/weak)
        
        forecast_data.append({
            'hour': hour,
            'timestamp': forecast_time.isoformat(),
            'flare_probability': round(flare_prob, 4),
            'no_flare_probability': round(no_flare_prob, 4),
            'flare_class': flare_class
        })
    
    # Create DataFrame
    df = pd.DataFrame(forecast_data)
    
    # Save to CSV
    output_path = '/Users/sudeepaweerasena/Desktop/suryamodel/solar_flare_forecast_24h.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Generated 24-hour forecast:")
    print(f"   File: {output_path}")
    print(f"   Hours: {len(df)}")
    print(f"\n📊 Forecast Summary:")
    print(f"   Average probability: {df['flare_probability'].mean()*100:.2f}%")
    print(f"   Peak probability: {df['flare_probability'].max()*100:.2f}% (hour {df.loc[df['flare_probability'].idxmax(), 'hour']})")
    print(f"   Risk level: ", end='')
    
    max_prob = df['flare_probability'].max()
    if max_prob > 0.7:
        print("🔴 HIGH RISK")
    elif max_prob > 0.4:
        print("🟡 MODERATE RISK")
    else:
        print("🟢 LOW RISK")
    
    print(f"\n💾 Saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    df = generate_24hour_forecast()
    print("\n📋 First 5 hours:")
    print(df.head().to_string(index=False))
