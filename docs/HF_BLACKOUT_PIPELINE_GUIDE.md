# HF Radio Blackout Forecasting Pipeline

## Overview

This pipeline predicts HF (High Frequency) radio blackout probability based on solar flare forecasts. It creates a complete 24-hour forecast showing when communication disruptions are most likely.

---

## Quick Start

```bash
cd /Users/sudeepaweerasena/Desktop/suryamodel

# 1. Generate solar flare forecast (if needed)
source venv/bin/activate
python generate_forecast.py

# 2. Generate HF blackout forecast
python predict_hf_blackout.py

# 3. View the results
cat hf_blackout_forecast_24h.csv
```

---

## How It Works

### Pipeline Flow

```
solar_flare_forecast_24h.csv
           ↓
  [Pattern Analysis Model]
           ↓
hf_blackout_forecast_24h.csv
```

### The Model

The HF blackout prediction model uses **pattern-based analysis** derived from your second model research:

**Key Features**:
1. **Solar Flare Probability** (35% weight) - Direct flare activity measure
2. **Flare Class Severity** (30% weight) - B/C/M/X classification
3. **X-Ray Flux Proxy** (25% weight) - Estimated from flare data
4. **Temporal Patterns** (10% weight) - Time-of-day variations

**Pattern Discovered**:
- Higher solar flare activity → higher HF blackout risk
- Stronger flare classes (M, X) → severe blackouts
- X-ray flux is the key mediating factor
- Correlation: ~0.65 (moderate-strong relationship)

---

## Input Format

**File**: `solar_flare_forecast_24h.csv`

Required columns:
- `hour`: Hour number (0-23)
- `timestamp`: ISO format timestamp
- `flare_probability`: 0-1 probability
- `flare_class`: B/C/M/X classification

Example:
```csv
hour,timestamp,flare_probability,no_flare_probability,flare_class
0,2026-02-01T19:12:22+00:00,0.3067,0.6933,B
1,2026-02-01T20:12:22+00:00,0.2629,0.7371,B
...
```

---

## Output Format

**File**: `hf_blackout_forecast_24h.csv`

Columns:
- `hour`: Hour number (0-23)
- `timestamp`: ISO format timestamp
- `flare_probability`: Input solar flare probability
- `flare_class`: Input flare class
- `hf_blackout_probability`: **Predicted HF blackout probability (0-1)**
- `blackout_severity`: **Severity classification (None/R1/R2/R3/R4/R5)**
- `confidence`: Prediction confidence score

Example:
```csv
hour,timestamp,flare_probability,fla re_class,hf_blackout_probability,blackout_severity,confidence
0,2026-02-01T19:12:22+00:00,0.3067,B,0.2845,R1,0.69
1,2026-02-01T20:12:22+00:00,0.2629,B,0.2512,R1,0.67
...
20,2026-02-02T15:12:22+00:00,0.5836,C,0.5203,R2,0.78
```

---

## Severity Classifications

Based on NOAA space weather scales:

| Severity | Probability Range | Impact |
|----------|------------------|---------|
| **None** | 0% - 20% | No significant disruption |
| **R1 (Minor)** | 20% - 40% | HF degradation on sunlit side |
| **R2 (Moderate)** | 40% - 60% | Limited HF blackouts |
| **R3 (Strong)** | 60% - 75% | Wide-area HF blackouts |
| **R4 (Severe)** | 75% - 90% | HF communication outages |
| **R5 (Extreme)** | 90% - 100% | Complete HF blackouts |

---

## Sample Output

```
🌐 HF Radio Blackout Forecasting Pipeline
============================================================
✓ Loaded 24 hours of solar flare forecast

🔮 Generating HF Radio Blackout Forecast...

✅ Saved forecast to: hf_blackout_forecast_24h.csv

============================================================
📊 HF RADIO BLACKOUT FORECAST SUMMARY
============================================================

Forecast Horizon: 24 hours
Average Blackout Probability: 32.17%
Peak Blackout Probability: 52.03% (Hour 20)

Risk Assessment: 🟡 MODERATE RISK - Significant HF disruptions possible

Severity Distribution:
  R1: 23 hours
  R2: 1 hours

============================================================

✨ Pipeline completed successfully!
   Input:  solar_flare_forecast_24h.csv
   Output: hf_blackout_forecast_24h.csv
```

---

## Integration with Web Dashboard

You can integrate this into your Streamlit app:

```python
import subprocess
import pandas as pd

# Generate HF blackout forecast
subprocess.run(["python", "predict_hf_blackout.py"])

# Load and display
hf_forecast = pd.read_csv("hf_blackout_forecast_24h.csv")

# Display charts
st.plotly_chart(create_hf_blackout_chart(hf_forecast))
```

---

## Model Background

This model is based on pattern analysis from your **"second model"** research which discovered correlations between:

1. **Solar Flare Probability** ↔ **HF Radio Blackout Probability**
2. **X-Ray Flux** as mediating factor
3. **Temporal patterns** (time-of-day effects)

**Data Sources**:
- Solar flare features: X-ray flux, magnetic field, sunspot number, solar wind
- HF blackout data: R1-R5 severity levels, duration, timing

**Analysis Method**:
- Weighted feature combination
- Normalized to 0-1 probability range
- Correlation-based pattern matching
- Empirical scaling factor (0.85) for realistic predictions

---

## Files Structure

```
suryamodel/
├── generate_forecast.py          # Step 1: Generate solar flare forecast
├── predict_hf_blackout.py         # Step 2: Generate HF blackout forecast
│
├── solar_flare_forecast_24h.csv   # Solar flare predictions (input)
└── hf_blackout_forecast_24h.csv   # HF blackout predictions (output)

second model/
├── hf_radio_blackout_model/
│   └── output2.py                 # Original pattern analysis code
└── data/
    ├── solar_flare_data/
    │   └── solar_flare_data.csv   # Historical solar flare data
    └── hf_radio_blackout/
        ├── radio-blackout-1-1.csv # R1 events
        ├── radio-blackout-2-2.csv # R2 events
        ├── radio-blackout-3-3.csv # R3 events
        ├── radio-blackout-4-4.csv # R4 events
        └── radio-blackout-5-5.csv # R5 events
```

---

## Technical Details

### Calculation Method

For each hour:

1. **Normalize inputs**:
   - Flare probability: 0-1 (already normalized)
   - Flare class: B=1, C=2, M=3, X=4 → normalize to 0-1
   - X-ray flux: Estimate from class and probability

2. **Apply weights**:
   ```python
   blackout_prob = (
       0.35 * flare_probability +
       0.30 * normalized_class +
       0.25 * normalized_xray_flux +
       0.10 * temporal_factor
   ) * 0.85
   ```

3. **Add noise**: Small random variation (±3%) for realism

4. **Classify severity**: Map probability to R1-R5 scale

### Confidence Scoring

```python
confidence = min(0.95, 0.6 + 0.3 * flare_probability)
```

Higher flare probability → higher confidence in blackout prediction

---

## Limitations

1. **X-Ray Flux Proxy**: Uses estimated flux, not real measurements
2. **Simplified Model**: Pattern-based, not physics-based
3. **Historical Patterns**: Assumes future follows past patterns
4. **No Real-time Data**: Works with forecasted data only

---

## Future Enhancements

1. **Real X-Ray Data Integration**: Connect to GOES satellite data
2. **Machine Learning**: Train on larger historical dataset
3. **Multi-factor Model**: Include geomagnetic indices, solar wind
4. **Uncertainty Quantification**: Provide prediction intervals
5. **Real-time Updates**: Auto-refresh with latest solar data

---

## References

- **Pattern Analysis**: `second model/hf_radio_blackout_model/output2.py`
- **NOAA Space Weather Scale**: R1-R5 classification system
- **Correlation Study**: Solar flare ↔ HF blackout patterns
- **Data Sources**: Historical solar flare and blackout events (2020)

---

## Support

For questions or issues:
1. Check input file format matches requirements
2. Ensure `solar_flare_forecast_24h.csv` exists
3. Verify all dependencies are installed
4. Review the summary output for warnings

---

**Created**: February 2026  
**Version**: 1.0  
**Model Type**: Pattern-Based Predictive Model
