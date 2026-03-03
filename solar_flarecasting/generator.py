"""
Generate a 24-hour solar flare forecast using trained ML models.

Uses pkl model files (RandomForestRegressor for probability, RandomForestClassifier
for flare class) with real-time NOAA SWPC data as input features. Falls back to
synthetic baseline data if NOAA APIs are unreachable.
"""

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import urllib.request
import json

# ---------------------------------------------------------------------------
# Model loading (cached at module level)
# ---------------------------------------------------------------------------

_models_cache = {}

def _get_model_dir():
    """Return the directory where pkl files are stored."""
    return os.path.dirname(os.path.abspath(__file__))

def _load_models():
    """Load all pkl artefacts once and cache them."""
    if _models_cache:
        return _models_cache

    model_dir = _get_model_dir()
    files = {
        'probability_model': 'solar_flare_probability_model.pkl',
        'class_model': 'solar_flare_class_model.pkl',
        'scaler': 'feature_scaler.pkl',
        'label_encoder': 'label_encoder.pkl',
        'feature_columns': 'feature_columns.pkl',
    }
    for key, fname in files.items():
        path = os.path.join(model_dir, fname)
        with open(path, 'rb') as f:
            _models_cache[key] = pickle.load(f)

    return _models_cache


# ---------------------------------------------------------------------------
# NOAA SWPC data fetching
# ---------------------------------------------------------------------------

_NOAA_URLS = {
    'xray':   'https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json',
    'kp':     'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
    'wind':   'https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json',
    'proton': 'https://services.swpc.noaa.gov/json/goes/primary/integral-protons-1-day.json',
}

def _fetch_json(url, timeout=10):
    """Fetch JSON from a URL, returns None on failure."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Surya2Boomi/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _fetch_noaa_data():
    """
    Fetch recent solar data from NOAA SWPC.
    Returns a dict with scalar values for
    xray_flux_short, xray_flux_long, magnetic_field,
    sunspot_number, solar_wind_speed, proton_flux.
    Returns None if all fetches fail.
    """
    data = {}
    success = False

    # --- X-ray flux ---
    xray = _fetch_json(_NOAA_URLS['xray'])
    if xray and len(xray) > 0:
        latest = xray[-1]
        data['xray_flux_short'] = float(latest.get('flux', 1e-6) or 1e-6)
        data['xray_flux_long']  = float(latest.get('flux', 1e-6) or 1e-6)
        success = True
    else:
        data['xray_flux_short'] = 1e-6
        data['xray_flux_long']  = 1e-6

    # --- Geomagnetic (Kp) ---
    kp = _fetch_json(_NOAA_URLS['kp'])
    if kp and len(kp) > 1:
        # First row is header; use the latest data row
        latest = kp[-1]
        try:
            data['magnetic_field'] = float(latest[1])
        except (ValueError, IndexError):
            data['magnetic_field'] = 2.0
        success = True
    else:
        data['magnetic_field'] = 2.0

    # --- Solar wind ---
    wind = _fetch_json(_NOAA_URLS['wind'])
    if wind and len(wind) > 1:
        latest = wind[-1]
        try:
            data['solar_wind_speed'] = float(latest[1]) if latest[1] else 400.0
        except (ValueError, IndexError):
            data['solar_wind_speed'] = 400.0
        success = True
    else:
        data['solar_wind_speed'] = 400.0

    # --- Proton flux ---
    proton = _fetch_json(_NOAA_URLS['proton'])
    if proton and len(proton) > 0:
        latest = proton[-1]
        data['proton_flux'] = float(latest.get('flux', 1.0) or 1.0)
        success = True
    else:
        data['proton_flux'] = 1.0

    # --- Sunspot number (static/estimated) ---
    # NOAA doesn't expose a real-time hourly JSON for sunspot number,
    # so we use a reasonable estimate for the current solar cycle phase.
    data['sunspot_number'] = 120.0  # ~solar cycle 25 active phase estimate

    return data if success else None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_feature_row(base_data, hour_offset, base_time, feature_columns):
    """
    Build a single feature row (dict) for *hour_offset* hours in the future.

    The model expects 90 features:
      - 7 base measurements
      - 6 time features (hour, day, month, year, day_of_year, day_of_week)
      - lag features (1, 3, 6, 12, 24) × 7 variables = 35
      - rolling stats (mean_6h, std_6h, mean_24h, std_24h) × 7 variables = 28
      - change features (change_1h, change_6h) × 7 variables = 14

    Since we only have a single snapshot of current data, lag/rolling/change
    features are approximated with small noise around the base values.
    """
    rng = np.random.default_rng(seed=int(base_time.timestamp()) + hour_offset)

    forecast_time = base_time + timedelta(hours=hour_offset)

    row = {}

    # Base measurements with slight hourly variation
    base_vars = [
        'solar_flare_probability', 'xray_flux_short', 'xray_flux_long',
        'magnetic_field', 'sunspot_number', 'solar_wind_speed', 'proton_flux',
    ]

    # Use a synthetic initial solar_flare_probability (will be overwritten by model)
    base_data_ext = dict(base_data)
    base_data_ext['solar_flare_probability'] = 0.15  # placeholder

    # Hourly variation factor
    hour_factor = 1.0 + 0.05 * np.sin(2 * np.pi * (forecast_time.hour - 6) / 24)

    for var in base_vars:
        val = base_data_ext[var] * hour_factor * rng.uniform(0.90, 1.10)
        row[var] = val

    # Time features
    row['hour'] = forecast_time.hour
    row['day'] = forecast_time.day
    row['month'] = forecast_time.month
    row['year'] = forecast_time.year
    row['day_of_year'] = forecast_time.timetuple().tm_yday
    row['day_of_week'] = forecast_time.weekday()

    # Lag features: approximate as base ± small noise
    for var in base_vars:
        for lag in [1, 3, 6, 12, 24]:
            row[f'{var}_lag_{lag}'] = row[var] * rng.uniform(0.85, 1.15)

    # Rolling stats
    for var in base_vars:
        val = row[var]
        for window in ['6h', '24h']:
            row[f'{var}_rolling_mean_{window}'] = val * rng.uniform(0.92, 1.08)
            row[f'{var}_rolling_std_{window}']  = abs(val) * rng.uniform(0.01, 0.15)

    # Change features
    for var in base_vars:
        val = row[var]
        row[f'{var}_change_1h'] = val * rng.uniform(-0.10, 0.10)
        row[f'{var}_change_6h'] = val * rng.uniform(-0.20, 0.20)

    # Ensure we only return columns the model expects, in correct order
    return {col: row.get(col, 0.0) for col in feature_columns}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_24hour_forecast(save_csv=False):
    """
    Generate 24-hour forecast using trained ML models.

    Args:
        save_csv (bool): Whether to save the forecast to a CSV file.

    Returns:
        pd.DataFrame: Forecast with columns
            hour, timestamp, flare_probability, no_flare_probability, flare_class
    """
    # Load models
    models = _load_models()
    prob_model      = models['probability_model']
    class_model     = models['class_model']
    scaler          = models['scaler']
    label_encoder   = models['label_encoder']
    feature_columns = models['feature_columns']

    # Fetch real-time data (or fall back to synthetic)
    base_data = _fetch_noaa_data()
    if base_data is None:
        # Fallback synthetic baseline
        base_data = {
            'xray_flux_short': 1e-6,
            'xray_flux_long':  1e-6,
            'magnetic_field':  2.0,
            'sunspot_number':  120.0,
            'solar_wind_speed': 400.0,
            'proton_flux':     1.0,
        }

    base_time = datetime.now(timezone.utc)

    # Build feature matrix for 24 hours
    rows = []
    for hour in range(24):
        row = _build_feature_row(base_data, hour, base_time, feature_columns)
        rows.append(row)

    features_df = pd.DataFrame(rows, columns=feature_columns)

    # Scale features
    features_scaled = scaler.transform(features_df)

    # Predict probability
    flare_probs = prob_model.predict(features_scaled)
    flare_probs = np.clip(flare_probs, 0.0, 1.0)

    # Predict class
    class_encoded = class_model.predict(features_scaled)
    class_labels  = label_encoder.inverse_transform(class_encoded)

    # Map label-encoder classes to standard B/C/M/X notation
    _class_map = {
        'No Flare': 'B',
        'C-class':  'C',
        'M-class':  'M',
        'X-class':  'X',
    }

    forecast_data = []
    for hour in range(24):
        forecast_time = base_time + timedelta(hours=hour)
        prob = float(flare_probs[hour])
        raw_label = str(class_labels[hour])
        flare_class = _class_map.get(raw_label, raw_label[0].upper() if raw_label else 'B')

        forecast_data.append({
            'hour': hour,
            'timestamp': forecast_time.isoformat(),
            'flare_probability': round(prob, 4),
            'no_flare_probability': round(1.0 - prob, 4),
            'flare_class': flare_class,
        })

    df = pd.DataFrame(forecast_data)

    if save_csv:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'solar_flare_forecast_24h.csv')
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
    df = generate_24hour_forecast(save_csv=True)
    print("\n📋 First 5 hours:")
    print(df.head().to_string(index=False))
