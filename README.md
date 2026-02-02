# 🌞 Solar Flare Forecasting - Colab API Setup

A complete solution for running 24-hour solar flare forecasting using NASA IMPACT's pretrained Surya model in Google Colab, with remote API access from your local machine.

## 🎯 Overview

This project provides:
- **Google Colab Notebook**: Complete environment with data, models, and API server
- **Pretrained Models**: NASA IMPACT's Surya foundation model for solar flare forecasting
- **REST API**: Flask-based API with ngrok tunnel for remote access
- **Local Client**: Python script to request forecasts from anywhere
- **24-Hour Predictions**: Hourly solar flare probability forecasts

## 📁 Project Structure

```
suryamodel/
├── solar_flare_colab_api.ipynb    # Main Colab notebook (run this first)
├── local_api_client.py             # Local client script
└── README.md                       # This file
```

## 🚀 Quick Start

### Step 1: Setup Google Colab

1. **Upload the notebook** to Google Colab:
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `solar_flare_colab_api.ipynb`

2. **Enable GPU** (recommended):
   - Click `Runtime` → `Change runtime type`
   - Select `Hardware accelerator` → `GPU` (T4 or better)
   - Click `Save`

3. **Run all cells sequentially**:
   - Click `Runtime` → `Run all`
   - Or use `Ctrl/Cmd + F9`

4. **Wait for setup to complete** (~5-10 minutes):
   - Dependencies installation
   - Repository cloning
   - Data and model downloads (~2-3 GB)
   - Model loading

5. **Copy the ngrok URL and API Key**:
   - After Step 6 runs, you'll see output like:
   ```
   ════════════════════════════════════════════════════════════════════
   🌍 NGROK TUNNEL ACTIVE
   ════════════════════════════════════════════════════════════════════
   
   📡 Public URL: https://xxxx-xx-xxx-xxx.ngrok-free.app
   🔑 API Key: your_api_key_here
   
   ════════════════════════════════════════════════════════════════════
   ```
   - **Save both values** - you'll need them for the local client!

### Step 2: Setup Local Client

1. **Install Python dependencies** (if not already installed):
   ```bash
   pip install requests
   ```

2. **Configure the client**:
   - Open `local_api_client.py` in a text editor
   - Update the configuration at the top:
   ```python
   NGROK_URL = "https://your-actual-ngrok-url.ngrok-free.app"
   API_KEY = "your-actual-api-key-from-colab"
   ```

3. **Test the connection**:
   ```bash
   python local_api_client.py --check-health
   ```
   
   Expected output:
   ```
   ✅ API is healthy!
      Service: Solar Flare Forecasting API
      Version: 1.0
      Device: cuda
   ```

### Step 3: Get Forecasts

**Basic forecast** (display only):
```bash
python local_api_client.py --forecast
```

**Save forecast to JSON**:
```bash
python local_api_client.py --forecast --output forecast_results.json
```

**Save forecast to CSV**:
```bash
python local_api_client.py --forecast --output forecast_results.csv
```

**Check API status**:
```bash
python local_api_client.py --status
```

## 📊 Forecast Output Format

### Console Display
```
════════════════════════════════════════════════════════════════════════════════
🌞 SOLAR FLARE FORECAST - NEXT 24 HOURS
════════════════════════════════════════════════════════════════════════════════

📅 Generated: 2026-02-01T14:30:00+00:00
🔭 Horizon: 24 hours
🤖 Model: Surya Solar Flare Forecaster
💻 Device: cuda

📊 Hourly Predictions (24 hours):
────────────────────────────────────────────────────────────────────────────────
Hour |      Timestamp       | Flare % | Class |         Status
────────────────────────────────────────────────────────────────────────────────
   0 | 2026-02-01T14:30:00 |  45.32% |     C | 🟡 Moderate
   1 | 2026-02-01T15:30:00 |  43.21% |     C | 🟡 Moderate
   2 | 2026-02-01T16:30:00 |  38.45% |     B | 🟢 Low Risk
...
```

### JSON Output
```json
{
  "status": "success",
  "forecast_generated_at": "2026-02-01T14:30:00+00:00",
  "forecast_horizon": "24 hours",
  "forecasts": [
    {
      "hour": 0,
      "timestamp": "2026-02-01T14:30:00+00:00",
      "no_flare_probability": 0.5468,
      "flare_probability": 0.4532,
      "flare_class": "C"
    },
    ...
  ],
  "model_info": {
    "name": "Surya Solar Flare Forecaster",
    "version": "1.0",
    "device": "cuda"
  }
}
```

### CSV Output
```csv
hour,timestamp,no_flare_probability,flare_probability,flare_class
0,2026-02-01T14:30:00+00:00,0.5468,0.4532,C
1,2026-02-01T15:30:00+00:00,0.5679,0.4321,C
...
```

## 🔧 Advanced Usage

### Using Command-Line Arguments

You can override the configured URL and API key:

```bash
python local_api_client.py --forecast \
  --url https://new-ngrok-url.ngrok-free.app \
  --api-key new_api_key_123
```

### API Endpoints

The Colab server exposes three endpoints:

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Check if API is running |
| `/status` | GET | Yes | Get model and data status |
| `/forecast` | POST | Yes | Generate 24-hour forecast |

### Direct API Calls

You can also use `curl` or any HTTP client:

**Health check:**
```bash
curl https://your-ngrok-url.ngrok-free.app/health
```

**Get forecast:**
```bash
curl -X POST https://your-ngrok-url.ngrok-free.app/forecast \
  -H "X-API-Key: your-api-key"
```

## ⚠️ Important Notes

### Session Duration
- **Free Colab**: Sessions timeout after ~12 hours of inactivity
- **Colab Pro**: Longer sessions available
- The API is only available while the notebook is running
- If the session ends, you'll need to re-run the notebook and get a new ngrok URL

### Ngrok Limitations
- **Free ngrok**: Sessions expire after 2 hours
- **Ngrok account** (free): Register at [ngrok.com](https://ngrok.com) to remove time limits
- Add your ngrok auth token in the notebook (Step 6) to avoid timeouts:
  ```python
  ngrok.set_auth_token("your_ngrok_token_here")
  ```

### GPU Availability
- Free Colab has limited GPU hours per week
- If GPU is unavailable, the model will run on CPU (slower but functional)
- Consider Colab Pro for guaranteed GPU access

### Data Freshness
- The current implementation uses the most recent data from the downloaded dataset
- For real-time forecasts, you would need to integrate with live SDO data feeds
- The model predictions are based on the training data distribution

## 🐛 Troubleshooting

### "Connection error" when running local client

**Problem**: Cannot connect to the API

**Solutions**:
1. Verify the Colab notebook is still running
2. Check that the ngrok URL is correct (copy-paste from Colab output)
3. Ensure your firewall allows outgoing connections
4. Try re-running Step 6 in the Colab notebook to get a fresh URL

### "Authentication failed" error

**Problem**: API key is incorrect

**Solutions**:
1. Copy the API key exactly from the Colab output (Step 6)
2. Make sure there are no extra spaces or quotes
3. If you re-ran the notebook, you'll have a new API key

### "Model not found" or "Data not available"

**Problem**: Data download failed or incomplete

**Solutions**:
1. Re-run Step 3 (data download) in the Colab notebook
2. Check Colab's output for errors during download
3. Verify you have enough Colab disk space (use `!df -h`)
4. Try manually running: `!bash download_data.sh`

### Slow inference or timeout

**Problem**: Forecasts take too long to generate

**Solutions**:
1. Check if GPU is enabled (`Runtime` → `Change runtime type`)
2. Verify GPU is available: Run a cell with `!nvidia-smi`
3. Increase timeout in `local_api_client.py` (line with `timeout=60`)
4. Try Colab Pro for better GPU availability

### Ngrok URL expires

**Problem**: Ngrok tunnel disconnects after 2 hours

**Solutions**:
1. Create a free ngrok account at [ngrok.com](https://ngrok.com)
2. Get your auth token from the dashboard
3. Add it to the notebook in Step 6:
   ```python
   ngrok.set_auth_token("your_token_here")
   ```
4. Re-run Step 6 to create a new tunnel with longer duration

## 📚 Additional Resources

- **Surya Repository**: [NASA-IMPACT/Surya](https://github.com/NASA-IMPACT/Surya)
- **Solar Flare Tutorial**: See the original `solar_flare_tutorial.ipynb` in the Surya repo
- **Google Colab**: [colab.research.google.com](https://colab.research.google.com)
- **Ngrok**: [ngrok.com](https://ngrok.com)

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the Colab notebook output for errors
3. Check the Surya repository for updates or known issues

## 📄 License

This project uses the NASA IMPACT Surya model. Please refer to the [Surya repository](https://github.com/NASA-IMPACT/Surya) for license information.

## 🌟 Features Summary

✅ No local GPU required - runs entirely in Google Colab  
✅ Automated setup with pretrained models  
✅ Remote API access from any machine  
✅ Secure API key authentication  
✅ Multiple output formats (console, JSON, CSV)  
✅ 24-hour hourly forecasts  
✅ Easy-to-use command-line interface  

---

**Built with ❤️ using NASA IMPACT's Surya Foundation Model**
