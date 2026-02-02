import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from datetime import datetime
import numpy as np

# Import forecast modules
import generate_forecast
import predict_hf_blackout

# Page configuration
st.set_page_config(
    page_title="HF Blackout Forecast System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# NASA Space Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@500&family=Share+Tech+Mono&display=swap');
    
    /* Style Streamlit header to blend in instead of hiding it completely */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }
    
    /* Hide specific header elements but keep the sidebar button */
    header[data-testid="stHeader"] > div:not([data-testid="collapsedControl"]) {
        display: none !important;
    }
    
    /* Keep Deploy button and settings hidden */
    [data-testid="stToolbar"], 
    [data-testid="stDecoration"],
    header[data-testid="stHeader"] button:not([data-testid="collapsedControl"]) {
        display: none !important;
    }
    
    /* Show and style the sidebar collapse button */
    [data-testid="collapsedControl"] {
        display: flex !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
        background: rgba(0, 217, 255, 0.2) !important;
        border: 1px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background: rgba(0, 217, 255, 0.3) !important;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.5) !important;
        transform: scale(1.05) !important;
    }
    
    [data-testid="collapsedControl"] svg {
        color: #00d9ff !important;
        width: 1.5rem !important;
        height: 1.5rem !important;
    }
    
    /* Also target the alternative selector */
    button[kind="header"],
    [data-testid="baseButton-header"] {
        display: flex !important;
        position: fixed !important;
        top: 1rem !important;
        left: 1rem !important;
        z-index: 999999 !important;
        background: rgba(0, 217, 255, 0.2) !important;
        border: 1px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="header"]:hover,
    [data-testid="baseButton-header"]:hover {
        background: rgba(0, 217, 255, 0.3) !important;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.5) !important;
        transform: scale(1.05) !important;
    }
    
    button[kind="header"] svg,
    [data-testid="baseButton-header"] svg {
        color: #00d9ff !important;
    }
    
    /* Remove top padding from main block */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Main app background - Deep space */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
    }
    
    /* Subtle starfield background image instead of pseudo-elements */
    body {
        background-image: 
            radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.15), transparent),
            radial-gradient(1px 1px at 70% 60%, rgba(255,255,255,0.1), transparent),
            radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.12), transparent);
        background-size: 200% 200%;
    }
    
    /* Typography */
    * {
        font-family: 'Rajdhani', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    
    /* Header section */
    .main-header {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem 3rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 60px rgba(0, 217, 255, 0.1);
        border-bottom: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        color: #00d9ff;
        text-align: center;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        margin-top: 0.5rem;
        font-weight: 400;
        letter-spacing: 1px;
    }
    
    /* Glass morphism cards */
    [data-testid="stMetric"],
    .element-container div[data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 217, 255, 0.3);
        border-color: rgba(0, 217, 255, 0.5);
    }
    
    /* Metric styling */
    [data-testid="stMetricLabel"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00d9ff;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00d9ff 0%, #667eea 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 1px;
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.5);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #00d9ff;
        margin: 1rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 217, 255, 0.3);
        letter-spacing: 2px;
        text-transform: uppercase;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
    }
    
    /* Glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
    }
    
    .glass-card:hover {
        border-color: rgba(0, 217, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
    }
    
    /* Severity badges */
    .severity-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .badge-r1 { 
        background: #ffd700; 
        color: #0a0e27; 
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    .badge-r2 { 
        background: #ff8c00; 
        color: #0a0e27; 
        box-shadow: 0 0 10px rgba(255, 140, 0, 0.5);
    }
    .badge-r3 { 
        background: #ff4500; 
        color: white; 
        box-shadow: 0 0 10px rgba(255, 69, 0, 0.5);
    }
    .badge-r4, .badge-r5 { 
        background: #dc143c; 
        color: white; 
        box-shadow: 0 0 10px rgba(220, 20, 60, 0.5);
    }
    
    /* Current alert */
    .current-alert {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 140, 0, 0.1) 100%);
        border: 2px solid rgba(255, 215, 0, 0.4);
        box-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        animation: pulse-alert 2s infinite;
    }
    
    @keyframes pulse-alert {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.3); }
        50% { box-shadow: 0 0 40px rgba(255, 215, 0, 0.5); }
    }
    
    .severity-large {
        font-family: 'Orbitron', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        color: #ffd700;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.6);
        letter-spacing: 3px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.98), rgba(26, 31, 46, 0.98));
        border-right: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        animation: pulse-dot 1.5s infinite;
        margin-right: 8px;
    }
    
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* UTC Time display */
    .utc-time {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        color: #00d9ff;
        text-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        text-align: center;
        letter-spacing: 2px;
    }
    
    /* Mini cards for header */
    .metric-mini {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .metric-mini-label {
        font-family: 'Orbitron', sans-serif;
        font-size: 0.65rem;
        color: #8b9dc3;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-mini-value {
        font-family: 'Share Tech Mono', monospace;
        font-size: 1.2rem;
        color: #00d9ff;
        margin-top: 0.25rem;
    }
    
    /* Timeline items */
    .timeline-item {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid;
        border-radius: 4px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .timeline-item:hover {
        background: rgba(255, 255, 255, 0.05);
        transform: translateX(5px);
    }
    
    /* Hourly breakdown table */
    .hourly-row {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(0, 217, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.3s ease;
    }
    
    .hourly-row:hover {
        background: rgba(0, 217, 255, 0.05);
        border-color: rgba(0, 217, 255, 0.3);
    }
    
    /* Progress bars */
    .progress-bar-container {
        background: rgba(255, 255, 255, 0.1);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Custom scrollbar styling for the hourly breakdown */
    .hourly-breakdown-scroll::-webkit-scrollbar {
        width: 8px;
    }
    
    .hourly-breakdown-scroll::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    .hourly-breakdown-scroll::-webkit-scrollbar-thumb {
        background: rgba(0, 217, 255, 0.4);
        border-radius: 4px;
    }
    
    .hourly-breakdown-scroll::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 217, 255, 0.6);
    }
    
    /* For Firefox */
    .hourly-breakdown-scroll {
        scrollbar-width: thin;
        scrollbar-color: rgba(0, 217, 255, 0.4) rgba(255, 255, 255, 0.05);
    }
    
    /* Text colors */
    p, li, span {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton>button {
        background: linear-gradient(135deg, #00d9ff 0%, #667eea 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.9rem 1.5rem;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        box-shadow: 0 4px 16px rgba(0, 217, 255, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
    }
    
    section[data-testid="stSidebar"] .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 217, 255, 0.6);
        background: linear-gradient(135deg, #00d9ff 20%, #667eea 100%);
    }

    
    /* Data tables */
    .dataframe {
        background: rgba(26, 32, 44, 0.6) !important;
        border-radius: 8px;
        border: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    .dataframe thead tr th {
        background: rgba(10, 14, 39, 0.95) !important;
        color: #8b9dc3 !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 217, 255, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# Header with NASA-style design
st.markdown("""
<div class="main-header">
    <p style="font-size: 1.8rem; font-weight: 700; margin: 0; color: #00d9ff;">📡 HF BLACKOUT FORECAST</p>
    <p style="font-size: 1rem; font-weight: 400; margin: 0; color: #8b9dc3;">24-Hour Solar Activity Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Load data (using session state)
def get_current_forecasts():
    """Get current forecasts from session state or generate default"""
    if 'solar_df' not in st.session_state:
        st.session_state.solar_df = None
    if 'hf_df' not in st.session_state:
        st.session_state.hf_df = None
        
    return st.session_state.solar_df, st.session_state.hf_df

@st.cache_resource
def get_models():
    """Load and cache the models"""
    try:
        return predict_hf_blackout.load_models("radio_blackout_models.pkl")
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return None

def generate_forecasts():
    """Generate both forecasts in-memory and update session state"""
    try:
        with st.spinner("⚡ Generating solar flare forecast..."):
             # Generate solar data (in-memory, no CSV)
            solar_df = generate_forecast.generate_24hour_forecast(save_csv=False)
            
        with st.spinner("📡 Predicting HF radio blackouts..."):
            models = get_models()
            if models is None:
                return
                
            # Generate HF blackout forecast (in-memory, no CSV)
            hf_df = predict_hf_blackout.generate_hf_blackout_forecast(solar_df, models, save_csv=False)
        
        # Update session state
        st.session_state.solar_df = solar_df
        st.session_state.hf_df = hf_df
        
        st.success("✅ Forecasts generated successfully!")
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Sidebar (keeping your existing sidebar)
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h3 style="
            color: #00d9ff; 
            font-family: 'Orbitron', sans-serif; 
            font-size: 1rem; 
            letter-spacing: 2px;
            margin: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        ">
            ⚙️ CONTROL PANEL
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate forecast button with gradient styling
    if st.button("🔄 GENERATE NEW FORECAST", use_container_width=True, type="primary"):
        generate_forecasts()
        st.rerun()
    
    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
    
    # System status
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="
            color: #00d9ff; 
            font-family: 'Orbitron', sans-serif; 
            font-size: 0.85rem; 
            letter-spacing: 1.5px;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            📊 SYSTEM STATUS
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
            <span class="status-dot"></span>
            <strong style="color: #4ade80; font-size: 0.95rem;">OPERATIONAL</strong>
        </div>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem; line-height: 1.5;">
            All systems nominal<br>
            Last updated: Just now
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Data sources
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="
            color: #00d9ff; 
            font-family: 'Orbitron', sans-serif; 
            font-size: 0.85rem; 
            letter-spacing: 1.5px;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            📡 DATA SOURCES
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="padding: 1rem; margin-bottom: 1.5rem;">
        <strong style="color: #00d9ff; font-size: 0.95rem;">Surya Foundation Model</strong><br>
        <div style="color: rgba(255,255,255,0.6); font-size: 0.85rem; margin-top: 0.5rem; line-height: 1.6;">
            • 366M parameters<br>
            • Pattern-based analysis<br>
            • 24-hour horizon
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # About
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <div style="
            color: #00d9ff; 
            font-family: 'Orbitron', sans-serif; 
            font-size: 0.85rem; 
            letter-spacing: 1.5px;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            ℹ️ ABOUT
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="padding: 1rem;">
        <div style="color: rgba(255,255,255,0.7); font-size: 0.85rem; line-height: 1.6;">
            NASA-inspired space weather monitoring powered by AI-driven pattern recognition and historical correlation analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)


# Load data
solar_df, hf_df = get_current_forecasts()

if solar_df is None or hf_df is None:
    st.warning("⚠️ No forecast data available. Please generate forecasts using the control panel.")
    st.stop()

# UTC Time and Quick Metrics Header
current_time = datetime.utcnow()
st.markdown(f"""
<div class="utc-time">{current_time.strftime('%H:%M:%S')} UTC</div>
<div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1.5rem;">
    <span class="status-dot"></span>
    <span style="color: #4ade80; font-size: 0.85rem; font-weight: 600;">SYSTEM OPERATIONAL</span>
</div>
""", unsafe_allow_html=True)

# Quick metrics in header
metric_col1, metric_col2, metric_col3 = st.columns(3)

# Calculate metrics
avg_solar = solar_df['flare_probability'].mean() * 100
avg_hf = hf_df['hf_blackout_probability'].mean() * 100
confidence_avg = hf_df['confidence'].mean() * 100

with metric_col1:
    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-mini-label">Solar Activity</div>
        <div class="metric-mini-value">{solar_df.iloc[0]['flare_class']}-Class</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col2:
    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-mini-label">Active Alerts</div>
        <div class="metric-mini-value">3</div>
    </div>
    """, unsafe_allow_html=True)

with metric_col3:
    st.markdown(f"""
    <div class="metric-mini">
        <div class="metric-mini-label">Avg Confidence</div>
        <div class="metric-mini-value">{confidence_avg:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Dashboard Grid (2-column layout)
left_col, right_col = st.columns([2, 1])

# LEFT PANEL - Main Chart and Supporting Info
with left_col:
    # Current Alert Card (Compact)
    current_hour = 0
    current_severity = hf_df.iloc[current_hour]['blackout_severity']
    current_prob = hf_df.iloc[current_hour]['hf_blackout_probability'] * 100
    
    st.markdown(f"""
    <div class="current-alert" style="padding: 1rem;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="text-align: left;">
                <div style="font-size: 0.65rem; color: #8b9dc3; letter-spacing: 2px; margin-bottom: 0.25rem;">CURRENT ALERT STATUS</div>
                <div style="font-family: 'Orbitron'; font-size: 2rem; font-weight: 700; color: #ffd700; text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);">{current_severity}</div>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; color: #e8f4f8; font-size: 0.85rem;"><strong>NEXT 60 MINUTES</strong></p>
                <p style="margin: 0; color: #e8f4f8; font-size: 0.85rem;">Minor HF Radio Blackout Expected</p>
                <p style="margin: 0; color: #e8f4f8; font-size: 0.85rem;">Probability: {current_prob:.1f}%</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Time-Series Chart
    st.markdown('<div class="section-header">SOLAR FLARE & HF BLACKOUT PROBABILITY (24H)</div>', unsafe_allow_html=True)
    
    # Create Plotly chart with real data
    fig_main = go.Figure()
    
    # Flare Probability (orange/red)
    fig_main.add_trace(go.Scatter(
        x=solar_df['hour'],
        y=solar_df['flare_probability'] * 100,
        name='Flare Probability',
        mode='lines',
        line=dict(color='#ff6b35', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 53, 0.2)',
        hovertemplate='<b>Hour %{x}</b><br>Flare: %{y:.1f}%<extra></extra>'
    ))
    
    # HF Blackout Probability (cyan/blue)
    fig_main.add_trace(go.Scatter(
        x=hf_df['hour'],
        y=hf_df['hf_blackout_probability'] * 100,
        name='Blackout Probability',
        mode='lines',
        line=dict(color='#00d9ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 217, 255, 0.2)',
        hovertemplate='<b>Hour %{x}</b><br>HF Blackout: %{y:.1f}%<extra></extra>'
    ))
    
    fig_main.update_layout(
        height=350,
        plot_bgcolor='rgba(0, 0, 0, 0.3)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Rajdhani', size=11, color='#e8f4f8'),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.12,
            xanchor="right",
            x=1,
            bgcolor='rgba(26, 32, 44, 0.8)',
            bordercolor='rgba(0, 217, 255, 0.3)',
            borderwidth=1,
            font=dict(family='Orbitron', size=9)
        ),
        xaxis=dict(
            title='Hour',
            gridcolor='rgba(0, 217, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            color='#8b9dc3'
        ),
        yaxis=dict(
            title='Probability (%)',
            gridcolor='rgba(0, 217, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            color='#8b9dc3'
        ),
        hovermode='x unified',
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig_main, use_container_width=True, key='main_chart')

    
    # Flare Class Distribution (4 boxes)
    st.markdown('<div class="section-header" style="margin-top: 1.5rem;">PREDICTED FLARE CLASS DISTRIBUTION</div>', unsafe_allow_html=True)
    
    class_counts = solar_df['flare_class'].value_counts()
    total_hours = len(solar_df)
    
    flare_col1, flare_col2, flare_col3, flare_col4 = st.columns(4)
    
    class_colors = {
        'B': ('#4a90e2', 'rgba(74, 144, 226, 0.2)'),
        'C': ('#f5a623', 'rgba(245, 166, 35, 0.2)'),
        'M': ('#ff6b35', 'rgba(255, 107, 53, 0.2)'),
        'X': ('#e74c3c', 'rgba(231, 76, 60, 0.2)')
    }
    
    for col, cls in zip([flare_col1, flare_col2, flare_col3, flare_col4], ['B', 'C', 'M', 'X']):
        count = class_counts.get(cls, 0)
        percentage = (count / total_hours * 100) if total_hours > 0 else 0
        border_color, bg_color = class_colors[cls]
        
        with col:
            st.markdown(f"""
            <div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 8px; padding: 1rem; text-align: center;">
                <div style="font-family: 'Orbitron'; font-size: 1.5rem; color: {border_color}; margin-bottom: 0.25rem;">{cls}</div>
                <div style="font-family: 'Share Tech Mono'; font-size: 1rem; color: #e8f4f8;">{count} hrs</div>
                <div style="font-size: 0.75rem; color: #8b9dc3; margin-top: 0.25rem;">{percentage:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Peak Risk Periods (3 boxes)
    st.markdown('<div class="section-header" style="margin-top: 1.5rem;">PEAK RISK PERIODS</div>', unsafe_allow_html=True)
    
    # Get top 3 peak hours
    peak_hours_df = hf_df.nlargest(3, 'hf_blackout_probability')
    
    peak_col1, peak_col2, peak_col3 = st.columns(3)
    
    for col, (idx, row) in zip([peak_col1, peak_col2, peak_col3], peak_hours_df.iterrows()):
        timestamp = pd.to_datetime(row['timestamp'])
        time_str = timestamp.strftime('%H:%M')
        prob = row['hf_blackout_probability'] * 100
        severity = row['blackout_severity']
        
        severity_colors = {
            'R1': '#ffd700',
            'R2': '#ff8c00',
            'R3': '#ff4500',
            'R4': '#dc143c',
            'R5': '#dc143c'
        }
        
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align: center; border: 1px solid {severity_colors.get(severity, '#ffd700')}40; padding: 1rem;">
                <div style="font-family: 'Share Tech Mono'; color: #ff6b35; font-size: 1rem; margin-bottom: 0.5rem;">{time_str} UTC</div>
                <span class="severity-badge badge-{severity.lower()}">{severity}</span>
                <div style="font-size: 0.85rem; color: #8b9dc3; margin-top: 0.5rem;">{prob:.1f}% Risk</div>
            </div>
            """, unsafe_allow_html=True)

# RIGHT PANEL - Risk Assessment and Data Tables
with right_col:
    # Risk Assessment Section
    st.markdown('<div class="section-header" style="font-size: 1rem;">RISK ASSESSMENT</div>', unsafe_allow_html=True)
    
    # Highest Risk Period
    max_idx = hf_df['hf_blackout_probability'].idxmax()
    max_row = hf_df.iloc[max_idx]
    max_hour = int(max_row['hour'])
    # Format time from timestamp
    max_time_str = pd.to_datetime(max_row['timestamp']).strftime('%H:%M')
    max_prob = max_row['hf_blackout_probability'] * 100
    max_severity = max_row['blackout_severity']
    max_class = max_row['flare_class']
    
    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #ff6b35; padding: 1rem;">
        <div style="font-size: 0.65rem; color: #8b9dc3; letter-spacing: 1px; margin-bottom: 0.5rem;">⚠️ HIGHEST RISK PERIOD</div>
        <div style="font-family: 'Share Tech Mono'; color: #00d9ff; font-size: 1.2rem; margin-bottom: 0.5rem;">{max_time_str} UTC</div>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Severity: <strong>{max_severity}</strong></p>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Probability: <strong>{max_prob:.1f}%</strong></p>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Flare Class: <strong>{max_class}-Class</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Lowest Risk Period
    min_idx = hf_df['hf_blackout_probability'].idxmin()
    min_row = hf_df.iloc[min_idx]
    min_hour = int(min_row['hour'])
    min_time_str = pd.to_datetime(min_row['timestamp']).strftime('%H:%M')
    min_prob = min_row['hf_blackout_probability'] * 100
    min_severity = min_row['blackout_severity']
    min_class = min_row['flare_class']
    
    st.markdown(f"""
    <div class="glass-card" style="border-left: 3px solid #4ade80; padding: 1rem; margin-top: 1rem;">
        <div style="font-size: 0.65rem; color: #8b9dc3; letter-spacing: 1px; margin-bottom: 0.5rem;">🛡️ LOWEST RISK PERIOD</div>
        <div style="font-family: 'Share Tech Mono'; color: #00d9ff; font-size: 1.2rem; margin-bottom: 0.5rem;">{min_time_str} UTC</div>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Severity: <strong>{min_severity}</strong></p>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Probability: <strong>{min_prob:.1f}%</strong></p>
        <p style="margin: 0.15rem 0; color: #e8f4f8; font-size: 0.85rem;">Flare Class: <strong>{min_class}-Class</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hourly Breakdown with Scrollbar
    st.markdown('<div class="section-header" style="font-size: 1rem; margin-top: 1.5rem;">HOURLY BREAKDOWN</div>', unsafe_allow_html=True)
    
    # Build the complete HTML for hourly breakdown
    hourly_html = """
    <div class="hourly-breakdown-scroll" style="
        max-height: 400px; 
        overflow-y: auto; 
        overflow-x: hidden; 
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(0, 217, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
    ">
    """
    
    # Add all 24 hours
    for i in range(len(hf_df)):
        cls = solar_df.iloc[i]['flare_class']
        severity = hf_df.iloc[i]['blackout_severity']
        prob = hf_df.iloc[i]['hf_blackout_probability'] * 100
        
        # Severity color for progress bar
        severity_color = {
            'R1': '#ffd700',
            'R2': '#ff8c00',
            'R3': '#ff4500',
            'R4': '#dc143c',
            'R5': '#dc143c'
        }.get(severity, '#ffd700')
        
        # Flare class badge colors
        flare_colors = {
            'B': ('rgba(74, 144, 226, 0.8)', '#ffffff'),  # Blue background, white text
            'C': ('rgba(245, 166, 35, 0.8)', '#0a0e27'),  # Orange background, dark text
            'M': ('rgba(255, 107, 53, 0.8)', '#ffffff'),  # Red-orange background, white text
            'X': ('rgba(231, 76, 60, 0.8)', '#ffffff')    # Red background, white text
        }
        flare_bg, flare_text = flare_colors.get(cls, ('#4a90e2', '#ffffff'))
        
        # Severity badge colors
        severity_colors_badge = {
            'R1': ('rgba(255, 215, 0, 0.8)', '#0a0e27'),   # Yellow background, dark text
            'R2': ('rgba(255, 140, 0, 0.8)', '#0a0e27'),   # Orange background, dark text
            'R3': ('rgba(255, 69, 0, 0.8)', '#ffffff'),    # Red-orange background, white text
            'R4': ('rgba(220, 20, 60, 0.8)', '#ffffff'),   # Red background, white text
            'R5': ('rgba(220, 20, 60, 0.8)', '#ffffff')    # Red background, white text
        }
        severity_bg, severity_text = severity_colors_badge.get(severity, ('#ffd700', '#0a0e27'))
        
        timestamp = pd.to_datetime(hf_df.iloc[i]['timestamp'])
        time_str = timestamp.strftime('%H:%M')
        
        hourly_html += f"""
        <div class="hourly-row" style="margin-bottom: 0.75rem;">
            <div style="display: flex; align-items: center; gap: 1rem; width: 100%;">
                <span style="font-family: 'Share Tech Mono'; color: #00d9ff; font-size: 0.9rem; min-width: 60px;">{time_str}</span>
                <span style="
                    font-size: 0.65rem; 
                    padding: 0.3rem 0.6rem; 
                    min-width: 30px; 
                    text-align: center;
                    background: {flare_bg};
                    color: {flare_text};
                    border-radius: 12px;
                    font-weight: 700;
                    font-family: 'Orbitron', sans-serif;
                    letter-spacing: 0.5px;
                    box-shadow: 0 0 8px {flare_bg};
                ">{cls}</span>
                <span style="
                    font-size: 0.65rem; 
                    padding: 0.3rem 0.6rem; 
                    min-width: 35px; 
                    text-align: center;
                    background: {severity_bg};
                    color: {severity_text};
                    border-radius: 12px;
                    font-weight: 700;
                    font-family: 'Orbitron', sans-serif;
                    letter-spacing: 0.5px;
                    box-shadow: 0 0 8px {severity_bg};
                ">{severity}</span>
                <div style="flex: 1;">
                    <div class="progress-bar-container" style="height: 6px; background: rgba(255, 255, 255, 0.1); border-radius: 4px; overflow: hidden;">
                        <div class="progress-bar" style="height: 100%; width: {prob}%; background: {severity_color}; border-radius: 4px; transition: width 0.3s ease;"></div>
                    </div>
                </div>
                <span style="font-family: 'Share Tech Mono'; color: #fff; font-size: 0.75rem; min-width: 45px; text-align: right;">{prob:.1f}%</span>
            </div>
        </div>
        """
    
    hourly_html += "</div>"
    
    # Use components.html for better rendering of large HTML
    import streamlit.components.v1 as components
    components.html(hourly_html, height=420, scrolling=False)


# Footer
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**Solar Flare Forecast Model** v2.0")
with footer_col2:
    if solar_df is not None:
        st.markdown(f"**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
with footer_col3:
    st.markdown("**Powered by:** NASA NOAA Space Weather Data")