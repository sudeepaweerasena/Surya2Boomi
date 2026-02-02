#!/usr/bin/env python3
"""
Solar Flare Forecasting Web Dashboard

A fully automated web application that displays 24-hour solar flare forecasts
from the Colab API with interactive visualizations and real-time updates.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Solar Flare Forecasting Dashboard",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "config.json"

# Initialize session state
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'config' not in st.session_state:
    st.session_state.config = {
        'ngrok_url': '',
        'api_key': '',
        'auto_refresh': False,
        'refresh_interval': 300
    }

# Load configuration
def load_config():
    """Load configuration from file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                st.session_state.config = json.load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")

# Save configuration
def save_config():
    """Save configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(st.session_state.config, f, indent=2)
    except Exception as e:
        st.error(f"Error saving config: {e}")

# API Client Functions
def fetch_forecast(ngrok_url, api_key):
    """Fetch forecast from the Colab API"""
    try:
        headers = {'X-API-Key': api_key}
        response = requests.post(
            f"{ngrok_url.rstrip('/')}/forecast",
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'status': 'error',
                'message': f'API returned status {response.status_code}'
            }
    except requests.exceptions.ConnectionError:
        return {
            'status': 'error',
            'message': 'Cannot connect to Colab. Make sure the notebook is running.'
        }
    except requests.exceptions.Timeout:
        return {
            'status': 'error',
            'message': 'Request timed out. The model might be processing.'
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

def check_api_health(ngrok_url):
    """Check if the API is online"""
    try:
        response = requests.get(f"{ngrok_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Visualization Functions
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
        hovertemplate='<b>Hour %{x}</b><br>' +
                      'Probability: %{y:.2f}%<br>' +
                      '<extra></extra>'
    ))
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.1, line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    
    # Update layout
    fig.update_layout(
        title="24-Hour Solar Flare Probability Forecast",
        xaxis_title="Hour",
        yaxis_title="Flare Probability (%)",
        hovermode='x unified',
        height=400,
        template='plotly_white',
        showlegend=True,
        yaxis=dict(range=[0, 105])
    )
    
    return fig

def create_risk_gauge(avg_prob):
    """Create risk gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_prob * 100,
        title={'text': "Average Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Main App
def main():
    # Load configuration on startup
    load_config()
    
    # Header
    st.title("🌞 Solar Flare Forecasting Dashboard")
    st.markdown("Real-time 24-hour solar flare predictions powered by NASA IMPACT's Surya AI model")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Settings
        st.subheader("API Connection")
        ngrok_url = st.text_input(
            "Ngrok URL",
            value=st.session_state.config.get('ngrok_url', ''),
            placeholder="https://your-ngrok-url.ngrok-free.dev",
            help="The public URL from your Colab notebook Step 6"
        )
        
        api_key = st.text_input(
            "API Key",
            value=st.session_state.config.get('api_key', ''),
            type="password",
            help="The API key from your Colab notebook Step 5"
        )
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            st.session_state.config['ngrok_url'] = ngrok_url
            st.session_state.config['api_key'] = api_key
            save_config()
            st.success("Configuration saved!")
        
        st.divider()
        
        # Auto-refresh settings
        st.subheader("🔄 Auto-Refresh")
        auto_refresh = st.checkbox(
            "Enable Auto-Refresh",
            value=st.session_state.config.get('auto_refresh', False)
        )
        
        refresh_interval = st.slider(
            "Refresh Interval (minutes)",
            min_value=1,
            max_value=30,
            value=st.session_state.config.get('refresh_interval', 5)
        )
        
        st.session_state.config['auto_refresh'] = auto_refresh
        st.session_state.config['refresh_interval'] = refresh_interval
        
        st.divider()
        
        # Connection status
        st.subheader("📡 Connection Status")
        if ngrok_url and api_key:
            if check_api_health(ngrok_url):
                st.success("✅ Connected to Colab")
            else:
                st.error("❌ Cannot reach Colab")
                st.caption("Make sure Step 6 is running in your Colab notebook")
        else:
            st.warning("⚠️ Configuration needed")
    
    # Main content
    if not ngrok_url or not api_key:
        st.warning("👈 Please configure your API settings in the sidebar to get started")
        st.info("""
        **Quick Start:**
        1. Make sure your Google Colab notebook is running (Step 6)
        2. Copy the ngrok URL from Colab
        3. Copy the API key from Colab
        4. Enter both in the sidebar
        5. Click 'Save Configuration'
        6. Click 'Fetch Latest Forecast' below
        """)
        return
    
    # Control buttons
    col1, col2, col3 = st.columns([2, 2, 6])
    
    with col1:
        if st.button("🔄 Fetch Latest Forecast", type="primary"):
            with st.spinner("Fetching forecast from Colab..."):
                result = fetch_forecast(ngrok_url, api_key)
                if result.get('status') == 'success':
                    st.session_state.forecast_data = result
                    st.session_state.last_update = datetime.now()
                    st.success("Forecast updated!")
                else:
                    st.error(f"Error: {result.get('message', 'Unknown error')}")
    
    with col2:
        if st.session_state.forecast_data:
            # Export buttons
            forecasts_df = pd.DataFrame(st.session_state.forecast_data['forecasts'])
            
            csv_data = forecasts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"solar_flare_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Auto-fetch on load if configured
    if st.session_state.forecast_data is None and ngrok_url and api_key:
        with st.spinner("Loading initial forecast..."):
            result = fetch_forecast(ngrok_url, api_key)
            if result.get('status') == 'success':
                st.session_state.forecast_data = result
                st.session_state.last_update = datetime.now()
    
    # Display forecast data
    if st.session_state.forecast_data:
        data = st.session_state.forecast_data
        
        # Last update info
        if st.session_state.last_update:
            st.caption(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary metrics
        forecasts = data['forecasts']
        df = pd.DataFrame(forecasts)
        
        avg_prob = df['flare_probability'].mean()
        max_prob = df['flare_probability'].max()
        max_hour = df.loc[df['flare_probability'].idxmax(), 'hour']
        
        # Determine risk level
        if max_prob > 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
        elif max_prob > 0.4:
            risk_level = "🟡 MODERATE RISK"
            risk_color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Average Probability",
                f"{avg_prob*100:.2f}%",
                help="Average flare probability across 24 hours"
            )
        
        with col2:
            st.metric(
                "Peak Probability",
                f"{max_prob*100:.2f}%",
                help="Highest flare probability in the forecast period"
            )
        
        with col3:
            st.metric(
                "Peak Hour",
                f"Hour {int(max_hour)}",
                help="Hour with highest flare probability"
            )
        
        with col4:
            st.markdown(f"**Risk Assessment**")
            st.markdown(f"### {risk_level}")
        
        st.divider()
        
        # Visualization tabs
        tab1, tab2, tab3 = st.tabs(["📊 Forecast Chart", "📈 Risk Gauge", "📋 Data Table"])
        
        with tab1:
            # Main forecast chart
            fig = create_forecast_chart(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Hourly breakdown
            st.subheader("Hourly Breakdown")
            
            # Create a more detailed view
            df_display = df.copy()
            df_display['flare_probability_pct'] = (df_display['flare_probability'] * 100).round(2)
            df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Color code by risk
            def get_risk_color(prob):
                if prob > 70:
                    return '🔴'
                elif prob > 40:
                    return '🟡'
                else:
                    return '🟢'
            
            df_display['risk'] = df_display['flare_probability_pct'].apply(get_risk_color)
            
            # Display in columns
            cols = st.columns(4)
            for idx, row in df_display.iterrows():
                col_idx = idx % 4
                with cols[col_idx]:
                    with st.container():
                        st.markdown(f"**Hour {row['hour']}** {row['risk']}")
                        st.caption(f"{row['flare_probability_pct']:.2f}% - Class {row['flare_class']}")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk gauge
                gauge_fig = create_risk_gauge(avg_prob)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Risk distribution
                st.subheader("Risk Distribution")
                
                low_risk = len(df[df['flare_probability'] <= 0.4])
                moderate_risk = len(df[(df['flare_probability'] > 0.4) & (df['flare_probability'] <= 0.7)])
                high_risk = len(df[df['flare_probability'] > 0.7])
                
                dist_fig = go.Figure(data=[go.Pie(
                    labels=['Low Risk', 'Moderate Risk', 'High Risk'],
                    values=[low_risk, moderate_risk, high_risk],
                    marker=dict(colors=['lightgreen', 'yellow', 'red']),
                    hole=0.4
                )])
                
                dist_fig.update_layout(
                    title="Hours by Risk Level",
                    height=300
                )
                
                st.plotly_chart(dist_fig, use_container_width=True)
        
        with tab3:
            # Data table
            st.subheader("Complete Forecast Data")
            
            # Prepare display dataframe
            df_table = df.copy()
            df_table['flare_probability'] = (df_table['flare_probability'] * 100).round(2).astype(str) + '%'
            df_table['no_flare_probability'] = (df_table['no_flare_probability'] * 100).round(2).astype(str) + '%'
            df_table['timestamp'] = pd.to_datetime(df_table['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Select columns to display
            display_cols = ['hour', 'timestamp', 'flare_probability', 'flare_class', 'no_flare_probability']
            if 'data_timestamp' in df_table.columns:
                display_cols.append('data_timestamp')
            
            st.dataframe(
                df_table[display_cols],
                use_container_width=True,
                hide_index=True
            )
        
        # Model info
        with st.expander("ℹ️ Model Information"):
            model_info = data.get('model_info', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Details:**")
                st.write(f"- Name: {model_info.get('name', 'N/A')}")
                st.write(f"- Type: {model_info.get('model_type', 'N/A')}")
                st.write(f"- Version: {model_info.get('version', 'N/A')}")
            
            with col2:
                st.write("**Forecast Details:**")
                st.write(f"- Generated: {data.get('forecast_generated_at', 'N/A')}")
                st.write(f"- Horizon: {data.get('forecast_horizon', 'N/A')}")
                st.write(f"- Device: {model_info.get('device', 'N/A')}")
    
    else:
        st.info("👆 Click 'Fetch Latest Forecast' to get started!")
    
    # Auto-refresh logic
    if auto_refresh and ngrok_url and api_key:
        time.sleep(refresh_interval * 60)
        st.rerun()

if __name__ == "__main__":
    main()
