#!/bin/bash

# Solar Forecast Dashboard Launcher
echo "🌌 Launching Solar Forecast Dashboard..."
echo "========================================="
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Error: Virtual environment not found"
    echo "   Run: ./setup_local_env.sh first"
    exit 1
fi

# Generate forecasts if they don't exist
if [ ! -f "solar_flare_forecast_24h.csv" ] || [ ! -f "hf_blackout_forecast_24h.csv" ]; then
    echo "🔮 Generating initial forecasts..."
    python generate_forecast.py
    python predict_hf_blackout.py
fi

# Launch dashboard
echo "🚀 Starting dashboard..."
echo ""
echo "  📡 URL: http://localhost:8501"
echo "  🌌 Beautiful space-themed interface!"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

streamlit run dashboard.py
