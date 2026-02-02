#!/bin/bash

# Quick Setup Script for Solar Flare Forecasting Local Client
# This script helps you quickly configure the local API client

echo "🌞 Solar Flare Forecasting - Local Client Setup"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully!"
else
    echo "❌ Failed to install dependencies. Please check the error messages above."
    exit 1
fi

echo ""
echo "================================================"
echo "🎯 Next Steps:"
echo "================================================"
echo ""
echo "1. Open Google Colab and upload 'solar_flare_colab_api.ipynb'"
echo "2. Run all cells in the notebook"
echo "3. Copy the ngrok URL and API key from the output"
echo "4. Edit 'local_api_client.py' and update:"
echo "   - NGROK_URL"
echo "   - API_KEY"
echo "5. Test the connection:"
echo "   python3 local_api_client.py --check-health"
echo ""
echo "For full instructions, see README.md"
echo ""
