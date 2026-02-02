#!/bin/bash

# Solar Flare Forecasting Web App Startup Script

echo "🌞 Solar Flare Forecasting Dashboard"
echo "===================================="
echo ""

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements_web.txt
fi

echo ""
echo "🚀 Starting web application..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo ""
echo "⚙️  First-time setup:"
echo "  1. Enter your ngrok URL from Colab Step 6"
echo "  2. Enter your API key from Colab Step 5"
echo "  3. Click 'Save Configuration'"
echo "  4. Click 'Fetch Latest Forecast'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit
streamlit run app.py
