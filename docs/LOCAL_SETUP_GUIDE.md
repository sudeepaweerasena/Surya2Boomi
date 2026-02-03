# 🌞 Solar Flare Forecasting - LOCAL MODE Setup Guide

## Quick Start (3 Steps!)

### Step 1: Setup Environment
```bash
cd /Users/sudeepaweerasena/Desktop/suryamodel
./setup_local_env.sh
```
**Time**: ~5 minutes  
**What it does**: Creates virtual environment, installs PyTorch & dependencies

### Step 2: Download Models
```bash
source venv/bin/activate
python download_models.py
```
**Time**: ~10-20 minutes (depends on internet speed)  
**What it does**: Downloads ~5GB of model weights and data from Hugging Face

### Step 3: Run the System
```bash
./run_local.sh
```
**Done!** 🎉 Your browser will open to http://localhost:8501

---

## What You Get

### ✅ Fully Local System
- **No Colab needed** - Everything runs on your Mac
- **No internet after setup** - Works offline
- **Complete privacy** - All data stays on your machine
- **No session timeouts** - Run as long as you want

### ✅ Hardware Support
- **Apple Silicon (M1/M2/M3)**: Uses Metal Performance Shaders (MPS) GPU
- **Intel Mac**: Uses CPU (slower but works)
- **Automatic detection**: System picks the best option

### ✅ Data Management
- **All forecasts saved**: `outputs/forecasts/`
- **CSV & JSON formats**: Easy to analyze
- **Timestamped files**: Never overwrite old data
- **Latest symlink**: Quick access to most recent forecast

---

## Directory Structure

After setup, your directory will look like this:

```
suryamodel/
├── venv/                    # Virtual environment
├── Surya/                   # Cloned repository
├── models/
│   └── checkpoint.pth       # Model weights (~3GB)
├── data/
│   ├── scalers.yaml         # Data scaling factors
│   ├── config_infer.yaml    # Model configuration
│   └── validation/          # Sample data
├── outputs/
│   └── forecasts/           # Your saved forecasts
│       ├── forecast_*.csv
│       ├── forecast_*.json
│       └── latest.csv       # Symlink to latest
├── cache/                   # Model cache (optional)
│
├── setup_local_env.sh       # Initial setup script
├── download_models.py       # Model downloader
├── local_inference.py       # Inference engine
├── app_local.py             # Web dashboard
└── run_local.sh             # One-command launcher
```

---

## Usage

### Web Dashboard (Recommended)
```bash
./run_local.sh
```
- Opens at http://localhost:8501
- Click "Generate Forecast"
- View interactive charts
- Download CSV

### Command Line
```bash
source venv/bin/activate

# Generate 24-hour forecast
python local_inference.py --hours 24

# Save to specific location
python local_inference.py --hours 48 --format csv

# Test mode
python local_inference.py --test
```

---

## Performance

### First Run
- Model loading: ~30-60 seconds
- Forecast generation: ~30-60 seconds
- **Total**: ~1-2 minutes

### Subsequent Runs
- Model already loaded: instant
- Forecast generation: ~30 seconds
- **Total**: ~30 seconds

### Hardware Performance
| Hardware | Model Load | Forecast (24h) |
|----------|-----------|----------------|
| M1/M2/M3 (MPS) | ~10s | ~20s |
| Intel Mac (CPU) | ~30s | ~60s |

---

## Troubleshooting

### "Module not found" errors
**Solution**: Make sure virtual environment is activated
```bash
source venv/bin/activate
```

### Model download fails
**Solution**: Check internet connection, or login to Hugging Face
```bash
pip install huggingface-hub
huggingface-cli login
python download_models.py
```

### Web app won't start
**Solution**: Check if port 8501 is already in use
```bash
# Use different port
streamlit run app_local.py --server.port 8502
```

### Out of memory
**Solution**: The model needs ~5-8GB RAM. Close other applications.

---

## Advanced Configuration

### Changing Forecast Horizon
In the web app sidebar, adjust the slider (6-48 hours)

Or via command line:
```bash
python local_inference.py --hours 48
```

### Custom Output Directory
```python
from local_inference import SolarFlareForecaster

forecaster = SolarFlareForecaster()
# ... setup ...
forecast = forecaster.generate_forecast_24h()
forecaster.save_forecast(forecast, output_dir="./my_forecasts")
```

### Model Caching
First run is slow (model loading). Subsequent runs reuse the loaded model automatically.

---

## Comparison: Colab vs Local

| Feature | Google Colab | Local Mac |
|---------|--------------|-----------|
| Setup Time | 5 min | 30 min (one-time) |
| Session Duration | 2 hours (free) | Unlimited |
| Internet Required | Always | Only for setup |
| GPU | T4 (free tier) | M1/M2/M3 (if available) |
| Privacy | Data on Google | Data on your machine |
| Cost | Free (with limits) | Free (uses your hardware) |
| Startup | 2-3 min | 10-30 sec |
| Inference Speed | Fast (GPU) | Fast (MPS) / Medium (CPU) |

---

## FAQs

**Q: Do I need to keep Colab running?**  
A: No! Everything runs locally now.

**Q: Can I use this offline?**  
A: Yes, after the initial setup and model download.

**Q: How much disk space needed?**  
A: ~10GB total (models + data + dependencies)

**Q: What about updates?**  
A: Pull latest Surya repo:
```bash
cd Surya
git pull
```

**Q: Can I deploy this to a server?**  
A: Yes! Works on any Linux/Mac server with Python 3.8+

---

## Next Steps

### 1. Generate Your First Forecast
```bash
./run_local.sh
```

### 2. Explore the Data
Check `outputs/forecasts/` for saved CSV files

### 3. Customize
- Modify `data/config_infer.yaml` for different model settings
- Edit `app_local.py` to customize the web interface
- Create batch processing scripts using `local_inference.py`

---

## Support

### Common Issues
1. **Port already in use**: Another app is using port 8501
2. **Module not found**: Virtual environment not activated
3. **Out of memory**: Close other apps, model needs 5-8GB RAM

### Getting Help
- Check the error message carefully
- Look in `outputs/` for error logs
- Review `data/config_infer.yaml` for configuration issues

---

**Enjoy your fully local solar flare forecasting system!** 🌞✨

No cloud, no limits, complete control!
