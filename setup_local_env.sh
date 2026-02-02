#!/bin/bash

# Solar Flare Forecasting - Local Environment Setup Script
# This script sets up everything needed to run the system locally

set -e  # Exit on error

echo "🌞 Solar Flare Forecasting - Local Setup"
echo "========================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo "📋 Checking Python version..."
PYTHON_CMD="python3"
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Found Python $PYTHON_VERSION"

# Check if Python version is 3.8 or higher
REQUIRED_VERSION="3.8"
if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    echo "❌ Error: Python 3.8 or higher required"
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with appropriate backend
echo ""
echo "🔥 Installing PyTorch..."
if [[ $(uname -m) == 'arm64' ]]; then
    echo "   Detected Apple Silicon - installing PyTorch with MPS support"
    pip install torch torchvision torchaudio --quiet
else
    echo "   Detected Intel Mac - installing PyTorch for CPU"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# Install other dependencies
echo ""
echo "📚 Installing dependencies..."
pip install -r requirements_local.txt --quiet
echo "   ✅ All dependencies installed"

# Create directory structure
echo ""
echo "📁 Creating directory structure..."
mkdir -p models
mkdir -p data/validation
mkdir -p outputs/forecasts
mkdir -p cache
mkdir -p Surya
echo "   ✅ Directories created"

# Check if Surya repository exists
echo ""
echo "📥 Checking Surya repository..."
if [ ! -d "Surya/.git" ]; then
    echo "   Cloning Surya repository..."
    git clone https://github.com/NASA-IMPACT/Surya.git --quiet
    echo "   ✅ Repository cloned"
else
    echo "   ℹ️  Repository already exists"
fi

# Test imports
echo ""
echo "🧪 Testing installation..."
$PYTHON_CMD -c "
import torch
import transformers
import streamlit
import pandas
import plotly
print('   ✅ All imports successful')

# Check device availability
if torch.backends.mps.is_available():
    print('   🚀 Apple Silicon GPU (MPS) available')
elif torch.cuda.is_available():
    print('   🚀 NVIDIA GPU (CUDA) available')
else:
    print('   💻 Using CPU (no GPU detected)')
"

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Download models: python download_models.py"
echo "  3. Run the system: ./run_local.sh"
echo ""
