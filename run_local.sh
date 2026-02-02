#!/bin/bash

# Solar Flare Forecasting - Local Launcher
# One-command startup for the fully local system

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🌞 Solar Flare Forecasting - Local Mode"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: ./setup_local_env.sh"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Check if models are downloaded
if [ ! -f "models/checkpoint.pth" ]; then
    echo "⚠️  Models not found!"
    echo "   Downloading models... (this will take 10-20 minutes)"
    echo ""
    python download_models.py
    echo ""
fi

# Start the web app
echo "🚀 Starting web application..."
echo ""
echo "  📡 URL: http://localhost:8501"
echo "  🖥️  Running in LOCAL mode (no Colab needed!)"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

# Run streamlit
streamlit run app_local.py

