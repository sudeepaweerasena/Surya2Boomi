#!/usr/bin/env python3
"""
Solar Flare Forecasting Web Dashboard - LOCAL MODE

Runs entirely on your local machine using the local_inference.py engine.
No cloud, no Colab, no API calls - everything is local!
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add local inference to path
sys.path.insert(0, str(Path(__file__).parent))

# Import local inference engine
try:
    from local_inference import SolarFlareForecaster
except ImportError as e:
    st.error(f"❌ Error importing local_inference: {e}")
    st.info("Make sure you ran: python download_models.py")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Solar Flare Forecasting - Local",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'device_info' not in st.session_state:
    st.session_state.device_info = None

def initialize_forecaster():
    """Initialize the forecaster (done once)"""
    if st.session_state.forecaster is None:
        with st.spinner("Initializing forecaster..."):
            try:
                forecaster = SolarFlareForecaster()
                
                # Detect device
                device_name = forecaster.detect_device()
                st.session_state.device_info = device_name
                
                # Load config
                forecaster.load_config()
                
                # Load model
                forecaster.load_model()
                
                st.session_state.forecaster = forecaster
                st.session_state.model_loaded = True
                
                return True, None
            except Exception as e:
                return False, str(e)
    return True, None

def create_forecast_chart(df):
    """Create interactive forecast chart"""
    fig = go.Figure()
    
    # Add flare probability line
    fig.add_trace(go.Scatter(
        x=df['hour'],
        y=df['flare_probability'] * 100,
        mode='lines+markers',
        name='Flare Probability',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)',
        hovertemplate='<b>Hour %{x}</b><br>' +
                      'Probability: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low Risk")
    fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate Risk")
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="High Risk")
    
    # Update layout
    fig.update_layout(
        title="24-Hour Solar Flare Probability Forecast",
        xaxis_title="Hour",
        yaxis_title="Flare Probability (%)",
        hovermode='x unified',
        height=450,
        template='plotly_white',
        showlegend=True,
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def main():
    """Main app"""
    
    # Header
    st.title("🌞 Solar Flare Forecasting Dashboard")
    st.markdown("**LOCAL MODE** - Running entirely on your machine!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Information")
        
        # Initialize if needed
        if not st.session_state.model_loaded:
            with st.status("Loading model...", expanded=True) as status:
                st.write("Initializing forecaster...")
                success, error = initialize_forecaster()
                
                if success:
                    st.write(f"✅ Device: {st.session_state.device_info}")
                    st.write("✅ Model loaded successfully")
                    status.update(label="Model ready!", state="complete", expanded=False)
                else:
                    st.error(f"❌ Error: {error}")
                    status.update(label="Error loading model", state="error", expanded=True)
                    st.stop()
        
        # Display system info
        if st.session_state.model_loaded:
            st.success("✅ Model Loaded")
            st.info(f"🖥️ {st.session_state.device_info}")
        
        st.divider()
        
        # Forecast settings
        st.subheader("🔮 Forecast Settings")
        num_hours = st.slider("Forecast Horizon (hours)", 6, 48, 24)
        
        st.divider()
        
        # Output settings
        st.subheader("💾 Output Settings")
        auto_save = st.checkbox("Auto-save forecasts", value=True)
        save_format = st.radio("Save Format", ["Both", "CSV Only", "JSON Only"])
    
    # Main content
    if not st.session_state.model_loaded:
        st.info("👈 Initializing model... Please wait")
        return
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        if st.button("🔄 Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Generating {num_hours}-hour forecast..."):
                forecast = st.session_state.forecaster.generate_forecast_24h(num_hours=num_hours)
                
                if forecast['status'] == 'success':
                    st.session_state.forecast_data = forecast
                    
                    # Auto-save if enabled
                    if auto_save:
                        format_map = {"Both": "both", "CSV Only": "csv", "JSON Only": "json"}
                        saved_files = st.session_state.forecaster.save_forecast(
                            forecast, 
                            format=format_map[save_format]
                        )
                        
                        with col2:
                            st.success(f"Saved!")
                    
                    st.success("Forecast generated!")
                    st.rerun()
                else:
                    st.error(f"Error: {forecast.get('message', 'Unknown error')}")
    
    with col2:
        if st.session_state.forecast_data:
            # Export button
            forecasts_df = pd.DataFrame(st.session_state.forecast_data['forecasts'])
            csv_data = forecasts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Display forecast
    if st.session_state.forecast_data:
        data = st.session_state.forecast_data
        forecasts = data['forecasts']
        df = pd.DataFrame(forecasts)
        
        # Calculate metrics
        avg_prob = df['flare_probability'].mean()
        max_prob = df['flare_probability'].max()
        max_hour = df.loc[df['flare_probability'].idxmax(), 'hour']
        
        # Risk level
        if max_prob >0.7:
            risk_level = "🔴 HIGH RISK"
            risk_color = "#FF4B4B"
        elif max_prob > 0.4:
            risk_level = "🟡 MODERATE RISK"
            risk_color = "#FFA500"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "#00CC00"
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Probability", f"{avg_prob*100:.2f}%")
        
        with col2:
            st.metric("Peak Probability", f"{max_prob*100:.2f}%")
        
        with col3:
            st.metric("Peak Hour", f"Hour {int(max_hour)}")
        
        with col4:
            st.markdown(f"**Risk Assessment**")
            st.markdown(f"### {risk_level}")
        
        st.divider()
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["📊 Forecast Chart", "📋 Data Table", "ℹ️ Info"])
        
        with tab1:
            # Main chart
            fig = create_forecast_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Hourly breakdown
            st.subheader("Hourly Breakdown")
            cols = st.columns(6)
            
            for idx, row in df.iterrows():
                col_idx = idx % 6
                with cols[col_idx]:
                    prob = row['flare_probability'] * 100
                    
                    if prob > 70:
                        emoji = "🔴"
                    elif prob > 40:
                        emoji = "🟡"
                    else:
                        emoji = "🟢"
                    
                    st.metric(
                        f"Hour {row['hour']} {emoji}",
                        f"{prob:.1f}%",
                        delta=None,
                        help=f"Class {row['flare_class']}"
                    )
        
        with tab2:
            # Data table
            st.subheader("Complete Forecast Data")
            
            df_display = df.copy()
            df_display['flare_probability'] = (df_display['flare_probability'] * 100).round(2).astype(str) + '%'
            df_display['no_flare_probability'] = (df_display['no_flare_probability'] * 100).round(2).astype(str) + '%'
            df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(
                df_display[['hour', 'timestamp', 'flare_probability', 'flare_class']],
                use_container_width=True,
                hide_index=True
            )
        
        with tab3:
            # Model info
            st.subheader("Model Information")
            
            model_info = data.get('model_info', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Details:**")
                st.write(f"- Name: {model_info.get('name', 'N/A')}")
                st.write(f"- Type: {model_info.get('model_type', 'N/A')}")
                st.write(f"- Version: {model_info.get('version', 'N/A')}")
            
            with col2:
                st.write("**System Details:**")
                st.write(f"- Device: {model_info.get('device', 'N/A')}")
                st.write(f"- Generated: {data.get('forecast_generated_at', 'N/A')}")
                st.write(f"- Horizon: {data.get('forecast_horizon', 'N/A')}")
            
            # Output directory info
            st.divider()
            st.subheader("📁 Output Directory")
            
            output_dir = Path(__file__).parent / "outputs" / "forecasts"
            if output_dir.exists():
                files = list(output_dir.glob("forecast_*"))
                st.write(f"Location: `{output_dir}`")
                st.write(f"Saved forecasts: {len([f for f in files if f.suffix in ['.csv', '.json']])} files")
                
                if files:
                    st.write("\n**Recent files:**")
                    for f in sorted(files, reverse=True)[:5]:
                        st.code(f.name, language=None)
    
    else:
        st.info("👆 Click 'Generate Forecast' to create your first forecast!")
        
        # Show quickstart
        with st.expander("🚀 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Welcome to the Local Solar Flare Forecaster!
            
            This app runs **completely on your machine** - no cloud, no Colab needed!
            
            **How to use:**
            1. Adjust forecast horizon in the sidebar (6-48 hours)
            2. Click **"Generate Forecast"** button
            3. View interactive charts and data
            4. Download CSV or check saved files in `outputs/forecasts/`
            
            **Features:**
            - 🖥️ Runs on your Mac (Apple Silicon GPU or CPU)
            - 💾 All data saved locally
            - 🔒 Complete privacy - nothing leaves your machine
            - ⚡ Fast after initial model loading
            
            **Note:** The first forecast may take 30-60 seconds as the model initializes. Subsequent forecasts are much faster!
            """)

if __name__ == "__main__":
    main()
