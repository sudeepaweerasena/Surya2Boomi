#!/usr/bin/env python3
"""
HF Radio Blackout Forecasting Pipeline (ML Model-Based)

Takes solar flare forecast CSV as input and predicts HF radio blackout probability
for the next 24 hours using trained machine learning models.

Input: solar_flare_forecast_24h.csv (24-hour solar flare forecast)
Output: hf_blackout_forecast_24h.csv (24-hour HF radio blackout forecast)
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timezone
from pathlib import Path

# Configuration
INPUT_FILE = "../data/solar_flare_forecast_24h.csv" # Relative to script dir
OUTPUT_FILE = "../data/hf_blackout_forecast_24h.csv"
MODEL_FILE = "radio_blackout_models.pkl" # In same dir as script

def load_models(model_path):
    """Load trained ML models from pickle file"""
    try:
        with open(model_path, 'rb') as f:
            models = pickle.load(f)
        
        # Only print when running as script
        if __name__ == "__main__":
            print("✓ Loaded ML models successfully")
            print(f"  - Severity Model: {type(models['severity_model']).__name__}")
            print(f"  - Probability Model: {type(models['probability_model']).__name__}")
        
        return models
    except Exception as e:
        raise Exception(f"Failed to load models: {e}")

def load_solar_flare_forecast(filepath):
    """Load solar flare forecast CSV"""
    df = pd.read_csv(filepath)
    if __name__ == "__main__":
        print(f"✓ Loaded {len(df)} hours of solar flare forecast")
    return df

def prepare_features(solar_flare_df, flare_class_encoder):
    """
    Prepare features for ML model prediction
    
    The model expects exactly 2 features: flare_class_encoded and solar_flare_probability
    """
    df = solar_flare_df.copy()
    
    # Encode flare class using the trained encoder
    try:
        df['flare_class_encoded'] = flare_class_encoder.transform(df['flare_class'])
    except Exception as e:
        # If the encoder doesn't recognize the class, use a default mapping
        if __name__ == "__main__":
            print(f"⚠️  Warning: Using fallback encoding for flare classes")
        flare_class_mapping = {'B': 0, 'C': 1, 'M': 2, 'X': 3}
        df['flare_class_encoded'] = df['flare_class'].map(flare_class_mapping).fillna(0)
    
    # Create feature matrix with exact names and order matching the training data
    # Model expects: ['flare_class_encoded', 'solar_flare_probability']
    features = pd.DataFrame({
        'flare_class_encoded': df['flare_class_encoded'],
        'solar_flare_probability': df['flare_probability']
    })
    
    return features

def predict_with_ml_models(models, features):
    """
    Use trained ML models to predict HF blackout probability and severity
    
    Args:
        models: Dictionary containing trained models
        features: DataFrame with prepared features
        
    Returns:
        Tuple of (probabilities, severities)
    """
    severity_model = models['severity_model']
    probability_model = models['probability_model']
    severity_encoder = models['severity_encoder']
    
    # Predict probability (continuous value 0-1)
    probabilities = probability_model.predict(features)
    
    # Clip probabilities to valid range
    probabilities = np.clip(probabilities, 0, 1)
    
    # Predict severity (categorical: R1-R5 or None)
    severity_encoded = severity_model.predict(features)
    
    # Decode severity labels
    try:
        severities = severity_encoder.inverse_transform(severity_encoded)
    except Exception as e:
        if __name__ == "__main__":
            print(f"⚠️  Warning: Error decoding severities, using fallback classification")
        severities = classify_severity_from_probability(probabilities)
    
    return probabilities, severities

def classify_severity_from_probability(probabilities):
    """
    Fallback function to classify severity based on probability thresholds
    
    R1 (Minor): 0.2-0.4
    R2 (Moderate): 0.4-0.6
    R3 (Strong): 0.6-0.75
    R4 (Severe): 0.75-0.9
    R5 (Extreme): 0.9-1.0
    """
    severities = []
    for prob in probabilities:
        if prob < 0.2:
            severities.append('None')
        elif prob < 0.4:
            severities.append('R1')
        elif prob < 0.6:
            severities.append('R2')
        elif prob < 0.75:
            severities.append('R3')
        elif prob < 0.9:
            severities.append('R4')
        else:
            severities.append('R5')
    return np.array(severities)

def generate_hf_blackout_forecast(solar_flare_df, models, save_csv=False):
    """
    Generate 24-hour HF blackout forecast from solar flare forecast using ML models
    
    Args:
        solar_flare_df (pd.DataFrame): Input solar flare forecast data
        models (dict): Loaded ML models
        save_csv (bool): Whether to save forecast to CSV
        
    Returns:
        pd.DataFrame: HF forecast
    """
    
    if __name__ == "__main__":
        print("\n🔮 Generating HF Radio Blackout Forecast with ML Models...")
    
    # Make a copy
    forecast_df = solar_flare_df.copy()
    
    # Prepare features
    features = prepare_features(forecast_df, models['flare_class_encoder'])
    
    # Predict using ML models
    probabilities, severities = predict_with_ml_models(models, features)
    
    # Add predictions to dataframe
    forecast_df['hf_blackout_probability'] = probabilities
    forecast_df['blackout_severity'] = severities
    
    # Calculate confidence score (based on model certainty)
    # For now, we'll use a simple heuristic: higher probability = higher confidence
    forecast_df['confidence'] = forecast_df.apply(
        lambda row: min(0.95, 0.6 + 0.3 * row['flare_probability']), axis=1
    )
    
    # Select output columns
    output_df = forecast_df[[
        'hour',
        'timestamp',
        'flare_probability',
        'flare_class',
        'hf_blackout_probability',
        'blackout_severity',
        'confidence'
    ]].copy()
    
    # Save output if requested
    if save_csv:
        # Use a default filename if running as library, or proper one if script
        script_dir = Path(__file__).parent if __file__ else Path.cwd()
        output_path = script_dir / OUTPUT_FILE
        save_forecast(output_df, output_path)
        
        # Only print summary when saving (usually means script mode)
        print_summary(output_df)
    
    return output_df

def save_forecast(df, output_path):
    """Save forecast to CSV"""
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved forecast to: {output_path}")

def print_summary(df):
    """Print forecast summary statistics"""
    print("\n" + "="*60)
    print("📊 HF RADIO BLACKOUT FORECAST SUMMARY")
    print("="*60)
    
    avg_prob = df['hf_blackout_probability'].mean()
    max_prob = df['hf_blackout_probability'].max()
    peak_hour = df.loc[df['hf_blackout_probability'].idxmax(), 'hour']
    
    print(f"\nForecast Horizon: {len(df)} hours")
    print(f"Average Blackout Probability: {avg_prob*100:.2f}%")
    print(f"Peak Blackout Probability: {max_prob*100:.2f}% (Hour {int(peak_hour)})")
    
    # Risk assessment
    if max_prob > 0.75:
        risk = "🔴 HIGH RISK - Severe HF blackouts expected"
    elif max_prob > 0.4:
        risk = "🟡 MODERATE RISK - Significant HF disruptions possible"
    else:
        risk = "🟢 LOW RISK - Minor HF disruptions only"
    
    print(f"\nRisk Assessment: {risk}")
    
    # Severity distribution
    print(f"\nSeverity Distribution:")
    severity_counts = df['blackout_severity'].value_counts().sort_index()
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count} hours")
    
    # Peak risk periods
    high_risk_hours = df[df['hf_blackout_probability'] > 0.6]
    if len(high_risk_hours) > 0:
        print(f"\n⚠️  High Risk Hours ({len(high_risk_hours)} total):")
        for _, row in high_risk_hours.head(5).iterrows():
            print(f"  Hour {int(row['hour'])}: {row['hf_blackout_probability']*100:.1f}% ({row['blackout_severity']})")
    
    print("\n" + "="*60)

def main():
    """Main pipeline execution"""
    
    print("\n🌐 HF Radio Blackout Forecasting Pipeline (ML Model-Based)")
    print("="*60)
    
    # Paths
    script_dir = Path(__file__).parent
    input_path = script_dir / INPUT_FILE
    output_path = script_dir / OUTPUT_FILE
    model_path = script_dir / MODEL_FILE
    
    # Check if model file exists
    if not model_path.exists():
        print(f"\n❌ Error: Model file not found: {model_path}")
        print("   Please ensure radio_blackout_models.pkl is in the project directory.")
        return 1
    
    # Check if input file exists
    if not input_path.exists():
        print(f"\n❌ Error: Input file not found: {input_path}")
        print("   Please run generate_forecast.py first to create solar flare forecast.")
        return 1
    
    try:
        # Load ML models
        models = load_models(model_path)
        
        # Load solar flare forecast
        solar_flare_df = load_solar_flare_forecast(input_path)
        
        # Generate HF blackout forecast (save_csv=True for script mode)
        hf_blackout_df = generate_hf_blackout_forecast(solar_flare_df, models, save_csv=True)
        
        # Summary is printed inside generate_hf_blackout_forecast when save_csv=True
        
        print("\n✨ Pipeline completed successfully!")
        print(f"   Input:  {INPUT_FILE}")
        print(f"   Output: {OUTPUT_FILE}")
        print(f"   Model:  {MODEL_FILE}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
