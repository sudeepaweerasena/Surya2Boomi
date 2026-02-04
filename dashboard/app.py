import sys
import os

# Add parent directory to path to allow importing sibling modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone
import numpy as np

# Import forecast modules from new structure
from surya_forecasting import generator as generate_forecast
from pattern_identification import predictor as predict_hf_blackout
st.set_page_config(
    page_title="SURYA2BOOMI - HF Blackout Forecast System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
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
        padding: 1rem 1.5rem;
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
        # Resolve absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        # Model is in pattern_identification/radio_blackout_models.pkl
        model_path = os.path.join(parent_dir, 'pattern_identification', 'radio_blackout_models.pkl')
        
        return predict_hf_blackout.load_models(model_path)
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

if "auto_generated" not in st.session_state:
    generate_forecasts()
    st.session_state.auto_generated = True
    st.rerun()
# Load data
solar_df, hf_df = get_current_forecasts()

if solar_df is None or hf_df is None:
    # If no data, show warning but don't crash - though stop() is fine here
    if not st.session_state.get('auto_generated', False):
         # If distinct from auto-gen failure
        st.warning("⚠️ No forecast data available. Please generate forecasts using the control panel.")
        st.stop()
    else:
        st.stop() # Should have been handled by auto-gen logic

# Calculate metrics for header
current_time = datetime.utcnow()
avg_solar = solar_df['flare_probability'].mean() * 100
# avg_hf used later
avg_hf = hf_df['hf_blackout_probability'].mean() * 100 
confidence_avg = hf_df['confidence'].mean() * 100
current_solar_class = solar_df.iloc[0]['flare_class']

# CSS for Header and Button
st.markdown("""
<style>
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #00d9ff 0%, #0077ff 100%);
        border: none;
        color: #000;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        width: 45% !important;
        display: block !important;
        margin: -5px 0 0 auto !important; /* Right align effectively or use margin auto for center */
        padding: 0.2rem 0.4rem !important;
        font-size: 0.7rem !important;
        min-height: auto !important;
        height: auto !important;
        line-height: 1.2 !important;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 15px rgba(0, 217, 255, 0.6);
        color: #000;
        /* Removed padding override to prevent jumping */
    }
    div.stButton > button[kind="primary"]:active {
        color: #000;
        background: linear-gradient(90deg, #00c3e6 0%, #0066d9 100%);
    }
</style>
""", unsafe_allow_html=True)

# Terms and Information Dialog
@st.dialog("About & Information", width="large")
def show_info_dialog():
    st.markdown("""
    <style>
        /* Force dark mode for dialog - override browser light mode */
        [data-testid="stDialog"],
        [data-testid="stDialog"] *,
        [data-testid="stModal"],
        [data-testid="stModal"] * {
            color-scheme: dark !important;
        }
        
        /* Dialog backdrop/overlay */
        [data-testid="stModal"] {
            background: rgba(0, 0, 0, 0.85) !important;
        }
        
        /* Dialog container - outer */
        [data-testid="stDialog"] {
            background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%) !important;
        }
        
        /* Dialog inner container - all possible selectors */
        [data-testid="stDialog"] > div,
        [data-testid="stDialog"] > div > div,
        [data-testid="stDialog"] [role="dialog"],
        [data-testid="stDialog"] section {
            background: rgba(10, 14, 39, 0.98) !important;
            background-color: rgba(10, 14, 39, 0.98) !important;
            border: 1px solid rgba(0, 217, 255, 0.3) !important;
            box-shadow: 0 0 40px rgba(0, 217, 255, 0.2) !important;
        }
        
        /* Override any white backgrounds */
        [data-testid="stDialog"] div[class*="st"],
        [data-testid="stDialog"] section[class*="st"] {
            background-color: transparent !important;
        }
        
        /* Headers */
        [data-testid="stDialog"] h1, 
        [data-testid="stDialog"] h2, 
        [data-testid="stDialog"] h3,
        [data-testid="stDialog"] h4,
        [data-testid="stDialog"] h5,
        [data-testid="stDialog"] h6 {
            color: #00d9ff !important;
            font-family: 'Orbitron', sans-serif !important;
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.5) !important;
            background: transparent !important;
        }
        
        /* Text elements */
        [data-testid="stDialog"] p,
        [data-testid="stDialog"] li,
        [data-testid="stDialog"] span,
        [data-testid="stDialog"] strong,
        [data-testid="stDialog"] em {
            color: rgba(255, 255, 255, 0.85) !important;
            font-family: 'Rajdhani', sans-serif !important;
            background: transparent !important;
        }
        
        /* Markdown container */
        [data-testid="stDialog"] [data-testid="stMarkdown"],
        [data-testid="stDialog"] [data-testid="stMarkdownContainer"] {
            background: transparent !important;
            background-color: transparent !important;
        }
        
        /* Horizontal rules */
        [data-testid="stDialog"] hr {
            border-color: rgba(0, 217, 255, 0.3) !important;
            background: rgba(0, 217, 255, 0.3) !important;
        }
        
        /* Close button styling */
        [data-testid="stDialog"] button[aria-label="Close"] {
            color: #00d9ff !important;
            background: rgba(0, 217, 255, 0.15) !important;
            border: 1px solid rgba(0, 217, 255, 0.3) !important;
            border-radius: 4px !important;
        }
        
        [data-testid="stDialog"] button[aria-label="Close"]:hover {
            background: rgba(0, 217, 255, 0.3) !important;
            box-shadow: 0 0 10px rgba(0, 217, 255, 0.4) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📡 HF Blackout Forecast System
    
    **24-Hour Solar flare Activity Monitoring & HF Radio Blackout Prediction**
    
    ---
    
    #### 🌟 About This System
    
    This advanced forecasting system provides real-time predictions for High Frequency (HF) radio blackouts caused by solar flare activity. 
    By monitoring solar flare activity patterns, we can predict potential disruptions to radio communications across various frequency bands.
    
    #### 🔬 How It Works
    
    - **Solar Flare Detection**: Solar flare predictions generated using the Surya forecasting model
    - **HF Impact Analysis**: Predicts radio blackout severity (R1-R5) for the next 24 hours
    - **Machine Learning**: Uses advanced algorithms trained on historical data
    
    #### 📊 Understanding Solar Flare Class Levels
    
    - **B-Class (Minor)**: Small solar flare representing low-level solar activity
    - **C-Class (Weak)**: Modest solar flare indicating moderate but common solar activity
    - **M-Class (Moderate)**: Strong solar flare capable of causing significant space-weather disturbances
    - **X-Class (Severe/Extreme)**: Extremely powerful solar flare representing the highest level of solar activity

    #### 📊 Understanding Severity Levels
    
    - **R1 (Minor)**: Weak degradation of HF radio on sunlit side
    - **R2 (Moderate)**: Limited blackout of HF radio, loss of contact for tens of minutes
    - **R3 (Strong)**: Wide area blackout, HF radio contact lost for about an hour
    - **R4 (Severe)**: HF radio blackout on most of sunlit side for one to two hours
    - **R5 (Extreme)**: Complete HF radio blackout on entire sunlit side for several hours
    
    #### ⚠️ Important Terms & Disclaimers
    
    **Data Sources**: This system utilizes data from NOAA Space Weather Prediction Center and NASA Solar Dynamics Observatory.
    
    **Forecast Accuracy**: Predictions are based on statistical models and historical patterns. Actual conditions may vary. 
    This system should be used as a guidance tool, not as the sole basis for critical operational decisions.
    
    **Refresh Rate**: Forecasts are generated on-demand. Click the "Refresh" button to get the latest predictions.
    
    **Time Zone**: All times displayed are in UTC (Coordinated Universal Time).
    
    #### 👨‍💻 Developer Information
    
    **Project**: Surya2Boomi v1.0  
    **Developer**: Sudeepa Weerasena  
    **Purpose**: Educational and research
    
    #### 📧 Contact & Support
    
    For questions, feedback, or support, please contact.    
    mailto: sudeepa.20221986@iit.ac.lk
    
    ---
    
    *This system is provided for informational purposes only. Use at your own discretion.*
    """)

# New Header Layout
header_col1, header_col2 = st.columns([2.5, 1.2], gap="medium")

with header_col1:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1.5rem; height: 100%; padding-top: 1rem;">
        <div style="
            width: 64px;
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <span style="font-size: 2.5rem;">📡</span>
        </div>
        <div>
            <div style="font-family: 'Orbitron', sans-serif; font-size: 1.5rem; font-weight: 700; color: #e2e8f0; letter-spacing: 2px;">SURYA2BOOMI</div>
            <p style="
                font-family: 'Rajdhani', sans-serif;
                font-size: 1rem;
                font-weight: 500;
                margin: 0.2rem 0 0 0;
                color: #8b9dc3;
                letter-spacing: 4px;
                text-transform: uppercase;
            ">24-Hour HF Blackout Forecast</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with header_col2:
    # Time
    st.markdown(f"""
    <div style="text-align: right; margin-bottom: 0.5rem;">
        <div style="
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            color: #00d9ff;
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
            letter-spacing: 2px;
        ">{current_time.strftime('%H:%M:%S')} UTC</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    st.markdown(f"""
    <div style="display: flex; justify-content: flex-end; margin-bottom: 1rem;">
        <div style="
            border: 1px solid #4ade80;
            border-radius: 4px;
            padding: 0.2rem 0.5rem;
            background: rgba(74, 222, 128, 0.1);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            <div style="width: 6px; height: 6px; background-color: #4ade80; border-radius: 50%; box-shadow: 0 0 5px #4ade80;"></div>
            <span style="color: #e2e8f0; font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; letter-spacing: 2px;">SYSTEM OPERATIONAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Button
    if st.button("Refresh", type="primary", use_container_width=True):
        generate_forecasts()
        st.rerun()

st.markdown('<div style="height: 1px; background: rgba(255,255,255,0.1); margin-top: 0.5rem; margin-bottom: 1.5rem;"></div>', unsafe_allow_html=True)




# Main Dashboard Grid (2-column layout)
left_col, right_col = st.columns([2, 1], gap="large")

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
    severity_priority = {
    "R1": 1,
    "R2": 2,
    "R3": 3,
    "R4": 4,
    "R5": 5
    }
    hf_df['severity_score'] = hf_df['blackout_severity'].map(severity_priority)
    hf_df_sorted = hf_df.sort_values(
        by=['severity_score', 'hf_blackout_probability'],
        ascending=[False, False]   # R5 → R1, then highest probability
    )
    max_row = hf_df_sorted.iloc[0]

    max_hour = int(max_row['hour'])
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
    hf_df_sorted_lowest = hf_df.sort_values(
    by=['severity_score', 'hf_blackout_probability'],
    ascending=[True, True]   # R1 → R5, then lowest probability
    )
    min_row = hf_df_sorted_lowest.iloc[0]

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


# Professional Footer with functional info button
st.markdown("""
<style>
    /* Footer container styling */
    .footer-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(10, 14, 39, 0.95);
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0, 217, 255, 0.2);
        padding: 0.75rem 2rem;
        z-index: 999;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Add bottom padding to main content to prevent footer overlap */
    .main .block-container {
        padding-bottom: 4rem;
    }
    
    /* Footer button styling */
    button[data-testid="baseButton-secondary"].footer-info,
    button[kind="secondary"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
        border: 1px solid rgba(0, 217, 255, 0.4) !important;
        border-radius: 50% !important;
        width: 20px !important;
        height: 20px !important;
        min-width: 20px !important;
        min-height: 20px !important;
        padding: 0 !important;
        margin: 0 !important;
        margin-left: -150px !important;
        transition: all 0.3s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[data-testid="baseButton-secondary"].footer-info:hover,
    button[kind="secondary"]:hover {
        background: rgba(0, 217, 255, 0.2) !important;
        background-color: rgba(0, 217, 255, 0.2) !important;
        box-shadow: 0 0 10px rgba(0, 217, 255, 0.4) !important;
    }
    
    button[data-testid="baseButton-secondary"] p,
    button[kind="secondary"] p {
        margin: 0 !important;
        padding: 0 !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 0.65rem !important;
        font-weight: 700 !important;
        color: #00d9ff !important;
    }
</style>
""", unsafe_allow_html=True)

# Create footer with columns
footer_container = st.container()
with footer_container:
    st.markdown('<div class="footer-container">', unsafe_allow_html=True)
    footer_col1, footer_col2, footer_col3 = st.columns([1, 1.5, 1])
    
    with footer_col1:
        # Version with info button inline
        v_col, i_col = st.columns([3, 1], gap="small")
        with v_col:
            st.markdown("""
                <span style="font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: #fff; font-weight: 600;">
                    Surya2Boomi v1.0
                </span>
            """, unsafe_allow_html=True)
        with i_col:
            if st.button("ⓘ", key="footer_info_button", type="secondary"):
                show_info_dialog()

    
    with footer_col2:
        st.markdown(f"""
            <div style="text-align: center; font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: rgba(255, 255, 255, 0.7);">
                Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
            </div>
        """, unsafe_allow_html=True)
    
    with footer_col3:
        st.markdown("""
            <div style="text-align: right; font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: rgba(255, 255, 255, 0.8);">
                Built by: <span style="color: #00d9ff;">Sudeepa Weerasena</span>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)