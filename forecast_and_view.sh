#!/bin/bash

# All-in-One Forecast & Dashboard Launcher
# Generates forecasts and opens dashboard in one command

echo "🚀 Space Weather Forecast Pipeline"
echo "===================================="
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Step 1: Generate solar flare forecast
echo "☀️  Generating solar flare forecast..."
python generate_forecast.py

# Step 2: Generate HF blackout forecast
echo "📡 Predicting HF radio blackouts..."
python predict_hf_blackout.py

# Step 3: Launch dashboard
echo ""
echo "✅ Forecasts ready!"
echo "🌐 Launching dashboard..."
echo ""
echo "  URL: http://localhost:8501"
echo ""

streamlit run dashboard.py
