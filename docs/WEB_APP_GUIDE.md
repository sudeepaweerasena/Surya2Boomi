# 🌞 Solar Flare Forecasting Web Application

## Quick Start Guide

### 1. Start Your Colab Notebook

1. Open `solar_flare_colab_api.ipynb` in Google Colab
2. Run all cells up to Step 6
3. **Keep Step 6 running** - don't stop it!
4. Copy the **ngrok URL** and **API Key** displayed

### 2. Launch the Web Application

**Option A: Using the startup script (Easy)**
```bash
cd /Users/sudeepaweerasena/Desktop/suryamodel
./start_webapp.sh
```

**Option B: Manual start**
```bash
cd /Users/sudeepaweerasena/Desktop/suryamodel
streamlit run app.py
```

Your browser will automatically open to: **http://localhost:8501**

### 3. First-Time Configuration

1. In the **sidebar**, enter your:
   - **Ngrok URL** (from Colab Step 6)
   - **API Key** (from Colab Step 5)
2. Click **"Save Configuration"**
3. The app will automatically fetch the forecast!

### 4. That's It! 🎉

- The dashboard will **automatically load** the forecast when you open it
- Click **"Fetch Latest Forecast"** to refresh data
- Enable **"Auto-Refresh"** for automatic updates every few minutes
- Download data using the **"Download CSV"** button

---

## Features

### 📊 Interactive Dashboard
- **Real-time forecasts** - 24-hour solar flare predictions
- **Interactive charts** - Hover for details, zoom, pan
- **Risk gauges** - Visual risk assessment
- **Data tables** - Complete hourly breakdown

### 🔄 Automatic Updates
- **Auto-fetch on load** - No manual clicking needed!
- **Auto-refresh mode** - Configurable refresh interval
- **Connection status** - See if Colab is online

### 📥 Data Export
- **CSV Download** - Export forecast data
- **JSON Download** - Raw API response
- **Timestamped files** - Never overwrite old data

### ⚙️ Smart Configuration
- **Saved settings** - Enter URL/key once, saved forever
- **Quick reconnect** - Change Colab session easily
- **Persistent config** - Settings survive browser refresh

---

## Usage

### Daily Workflow

```bash
# 1. Start Colab (once per session)
# → Run Colab notebook Step 6
# → Keep it running

# 2. Open web app (anytime)
./start_webapp.sh

# That's it! The app handles everything else automatically.
```

### Updating Configuration

If Colab restarts or you get a new ngrok URL:

1. Copy new ngrok URL from Colab
2. Update in sidebar
3. Click "Save Configuration"
4. Done!

---

## Web App Screenshots

### Main Dashboard
Interactive 24-hour forecast with risk zones:
- 🟢 Green zone: Low risk (0-40%)
- 🟡 Yellow zone: Moderate risk (40-70%)
- 🔴 Red zone: High risk (70-100%)

### Summary Cards
- Average Probability
- Peak Probability
- Peak Hour
- Risk Assessment

### Data Table
Complete hourly forecast data with timestamps and probabilities

---

## Troubleshooting

### "Cannot connect to Colab"
**Solution**: Make sure Step 6 in Colab is still running
```
Go to Colab → Check if cell is active → Re-run if needed
```

### "Invalid or missing API key"
**Solution**: Update your API key in the sidebar
```
1. Copy new API key from Colab Step 5
2. Paste in sidebar
3. Save Configuration
```

### Web app won't start
**Solution**: Install dependencies
```bash
pip3 install -r requirements_web.txt
```

### Port already in use
**Solution**: Stop other Streamlit apps or use different port
```bash
streamlit run app.py --server.port 8502
```

---

## Advanced Features

### Auto-Refresh Configuration
Enable automatic forecast updates:
1. Check **"Enable Auto-Refresh"** in sidebar
2. Set refresh interval (1-30 minutes)
3. App will fetch new forecasts automatically

### Multiple Colab Sessions
Switch between different Colab sessions easily:
1. Update ngrok URL in sidebar
2. Update API key if changed
3. Save configuration
4. Fetch new forecast

---

## Technical Details

### Architecture
```
Browser (localhost:8501)
    ↓
Streamlit App (app.py)
    ↓
HTTP Request
    ↓
Ngrok Tunnel
    ↓
Google Colab (Flask API)
    ↓
Surya AI Model
```

### Files
- `app.py` - Main web application
- `config.json` - Saved configuration (auto-created)
- `requirements_web.txt` - Python dependencies
- `start_webapp.sh` - Startup script

### Dependencies
- Streamlit - Web framework
- Plotly - Interactive charts
- Pandas - Data processing
- Requests - API communication

---

## Next Steps

### Enhancements You Can Add

1. **Historical Tracking**
   - Store forecast history
   - Compare predictions over time
   - Trend analysis

2. **Alerts**
   - Email notifications for high-risk forecasts
   - Desktop notifications
   - SMS alerts via Twilio

3. **Advanced Visualizations**
   - Heatmaps
   - Time-series comparisons
   - Probability distributions

4. **Data Analysis**
   - Prediction accuracy tracking
   - Statistical summaries
   - Export to BI tools

---

## Support

### Common Questions

**Q: Do I need to keep Colab running?**  
A: Yes, the model runs in Colab. Keep Step 6 running for the API to work.

**Q: Can I deploy this online?**  
A: Yes! Deploy to Streamlit Cloud, Heroku, or any Python hosting. You'll need to deploy the model too or use a cloud GPU service.

**Q: How often should I refresh?**  
A: Depends on your needs. Solar conditions change slowly, so 5-15 minutes is reasonable.

**Q: Can multiple people use the web app?**  
A: Yes! Share your local URL (via ngrok for external access) or deploy to a server.

---

## License & Attribution

Built using:
- **NASA IMPACT Surya** - Foundation model
- **Streamlit** - Web framework
- **Plotly** - Visualizations

---

**Enjoy your automated solar flare forecasting dashboard!** 🌞✨
