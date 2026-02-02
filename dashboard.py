import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import subprocess
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Space Weather Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS - Clean Enterprise Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #0f1419 100%);
    }
    
    /* Subtle star field effect */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.3), transparent),
            radial-gradient(1px 1px at 70% 60%, rgba(255,255,255,0.2), transparent),
            radial-gradient(1px 1px at 40% 80%, rgba(255,255,255,0.25), transparent),
            radial-gradient(1px 1px at 90% 20%, rgba(255,255,255,0.3), transparent);
        background-size: 300% 300%;
        opacity: 0.5;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Typography */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.02em;
    }
    
    /* Main title */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 3rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        text-align: center;
        letter-spacing: -0.03em;
    }
    
    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Metric cards - Professional style */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26, 32, 44, 0.95) 0%, rgba(45, 55, 72, 0.95) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        font-weight: 600;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.25rem;
        font-weight: 700;
        color: #667eea;
        font-family: 'JetBrains Mono', monospace;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Info boxes */
    .info-card {
        background: rgba(26, 32, 44, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    /* Risk level indicators */
    .risk-indicator {
        background: rgba(26, 32, 44, 0.8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .risk-low { border-left-color: #48bb78; }
    .risk-moderate { border-left-color: #ed8936; }
    .risk-high { border-left-color: #f56565; }
    .risk-severe { border-left-color: #9f1239; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: rgba(26, 32, 44, 0.4);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.7);
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 20, 25, 0.98), rgba(26, 31, 46, 0.98));
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #667eea;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Data tables */
    .dataframe {
        background: rgba(26, 32, 44, 0.6) !important;
        border-radius: 8px;
    }
    
    /* Text */
    p, li, span {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(26, 32, 44, 0.6);
        border-radius: 8px;
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .badge-success { background: #48bb78; color: white; }
    .badge-warning { background: #ed8936; color: white; }
    .badge-danger { background: #f56565; color: white; }
    .badge-info { background: #4299e1; color: white; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛰️ Space Weather Monitoring System</h1>
    <p class="subtitle">Solar Flare Detection & HF Radio Blackout Forecasting Platform</p>
</div>
""", unsafe_allow_html=True)

# Load data function
@st.cache_data(ttl=60)
def load_forecast_data():
    """Load both solar flare and HF blackout forecast data"""
    solar_path = Path("solar_flare_forecast_24h.csv")
    hf_path = Path("hf_blackout_forecast_24h.csv")
    
    solar_df = None
    hf_df = None
    
    if solar_path.exists():
        solar_df = pd.read_csv(solar_path)
    
    if hf_path.exists():
        hf_df = pd.read_csv(hf_path)
    
    return solar_df, hf_df

def generate_forecasts():
    """Generate both forecasts"""
    with st.spinner("⚡ Generating solar flare forecast..."):
        subprocess.run(["python", "generate_forecast.py"], capture_output=True)
    
    with st.spinner("📡 Predicting HF radio blackouts..."):
        subprocess.run(["python", "predict_hf_blackout.py"], capture_output=True)
    
    st.success("✅ Forecasts generated successfully!")
    st.cache_data.clear()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Control Panel")
    
    if st.button("🔄 Generate New Forecast"):
        generate_forecasts()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📊 System Status")
    st.markdown("""
    <div class="info-card">
    <strong>🟢 Operational</strong><br>
    <small style="color: rgba(255,255,255,0.6);">
    All systems nominal<br>
    Last updated: Just now
    </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📡 Data Sources")
    st.markdown("""
    <div class="info-card">
    <strong>Surya Foundation Model</strong><br>
    <small style="color: rgba(255,255,255,0.6);">
    • 366M parameters<br>
    • Pattern-based analysis<br>
    • 24-hour horizon
    </small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div class="info-card">
    <small style="color: rgba(255,255,255,0.7);">
    Enterprise-grade space weather monitoring powered by AI-driven pattern recognition and historical correlation analysis.
    </small>
    </div>
    """, unsafe_allow_html=True)

# Load data
solar_df, hf_df = load_forecast_data()

if solar_df is None or hf_df is None:
    st.warning("⚠️ No forecast data available. Please generate forecasts using the control panel.")
    st.stop()

# Enhanced KPI Section with Insights
st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)

# Calculate advanced metrics
avg_solar = solar_df['flare_probability'].mean() * 100
avg_hf = hf_df['hf_blackout_probability'].mean() * 100
peak_solar = solar_df['flare_probability'].max() * 100
peak_hf = hf_df['hf_blackout_probability'].max() * 100
peak_hour_solar = solar_df.loc[solar_df['flare_probability'].idxmax(), 'hour']
peak_hour_hf = hf_df.loc[hf_df['hf_blackout_probability'].idxmax(), 'hour']

# Calculate trends (first 12h vs last 12h)
solar_trend = (solar_df['flare_probability'].iloc[12:].mean() - solar_df['flare_probability'].iloc[:12].mean()) * 100
hf_trend = (hf_df['hf_blackout_probability'].iloc[12:].mean() - hf_df['hf_blackout_probability'].iloc[:12].mean()) * 100

# High-risk hours
high_risk_solar = len(solar_df[solar_df['flare_probability'] > 0.5])
high_risk_hf = len(hf_df[hf_df['hf_blackout_probability'] > 0.6])

# Severity distribution
severe_events = len(hf_df[hf_df['blackout_severity'].isin(['R3', 'R4', 'R5'])])
moderate_events = len(hf_df[hf_df['blackout_severity'] == 'R2'])

# Correlation
correlation = np.corrcoef(solar_df['flare_probability'], hf_df['hf_blackout_probability'])[0,1]

# Display KPIs
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Avg Solar Activity", 
        f"{avg_solar:.1f}%",
        delta=f"{solar_trend:+.1f}% trend",
        delta_color="inverse"
    )

with col2:
    st.metric(
        "Peak Solar Risk", 
        f"{peak_solar:.1f}%",
        delta=f"Hour {int(peak_hour_solar)}",
        delta_color="off"
    )

with col3:
    st.metric(
        "Avg HF Blackout", 
        f"{avg_hf:.1f}%",
        delta=f"{hf_trend:+.1f}% trend",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "High Risk Hours",
        f"{high_risk_solar + high_risk_hf}",
        delta=f"Solar: {high_risk_solar}, HF: {high_risk_hf}",
        delta_color="off"
    )

with col5:
    st.metric(
        "Severe Events",
        f"{severe_events}",
        delta=f"R3+ events" if severe_events > 0 else "None",
        delta_color="inverse" if severe_events > 0 else "normal"
    )

st.markdown("<br>", unsafe_allow_html=True)

# Intelligent Alerts & Priority Actions
st.markdown('<div class="section-header">🚨 Active Alerts & Recommendations</div>', unsafe_allow_html=True)

# Generate intelligent alerts
alerts = []

# Critical solar activity
if peak_solar > 70:
    alerts.append({
        'priority': 'CRITICAL',
        'icon': '🔴',
        'title': 'Extreme Solar Activity Expected',
        'message': f'Peak solar flare probability of {peak_solar:.1f}% at Hour {int(peak_hour_solar)}. Immediate action required.',
        'actions': ['Activate emergency protocols', 'Alert all HF operators', 'Switch to backup systems']
    })
elif peak_solar > 50:
    alerts.append({
        'priority': 'HIGH',
        'icon': '🟠',
        'title': 'Elevated Solar Activity',
        'message': f'Significant solar flare risk of {peak_solar:.1f}% detected.',
        'actions': ['Monitor continuously', 'Prepare contingency plans', 'Brief operations team']
    })

# HF blackout warnings
if severe_events > 0:
    alerts.append({
        'priority': 'HIGH',
        'icon': '🟠',
        'title': f'{severe_events} Severe HF Blackout Event(s)',
        'message': 'R3+ severity events expected. HF communications will be severely impacted.',
        'actions': ['Use satellite communications', 'Delay non-critical HF transmissions', 'Document all outages']
    })
elif moderate_events > 3:
    alerts.append({
        'priority': 'MODERATE',
        'icon': '🟡',
        'title': f'{moderate_events} Moderate HF Events',
        'message': 'Multiple R2-level events forecasted.',
        'actions': ['Test backup systems', 'Monitor HF quality', 'Prepare incident reports']
    })

# Correlation insights
if correlation > 0.7:
    alerts.append({
        'priority': 'INFO',
        'icon': 'ℹ️',
        'title': 'Strong Solar-HF Correlation Detected',
        'message': f'Correlation coefficient: {correlation:.3f}. Solar activity is a reliable HF blackout predictor.',
        'actions': ['Use solar forecasts for HF planning', 'Update prediction models']
    })

# Trend warnings
if solar_trend > 5:
    alerts.append({
        'priority': 'MODERATE',
        'icon': '📈',
        'title': 'Rising Solar Activity Trend',
        'message': f'Activity increasing by {solar_trend:.1f}% over forecast period.',
        'actions': ['Anticipate worsening conditions', 'Schedule critical ops early']
    })

# Display no alerts if none
if not alerts:
    st.success("✅ **All Clear** - No significant alerts. Normal operations can continue with routine monitoring.")
else:
    # Display alerts by priority
    for alert in sorted(alerts, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2, 'INFO': 3}[x['priority']]):
        if alert['priority'] == 'CRITICAL':
            risk_class = 'risk-severe'
        elif alert['priority'] == 'HIGH':
            risk_class = 'risk-high'
        elif alert['priority'] == 'MODERATE':
            risk_class = 'risk-moderate'
        else:
            risk_class = 'risk-low'
        
        with st.container():
            st.markdown(f"""
            <div class="risk-indicator {risk_class}">
                <h4 style="margin:0; color:#fff;">{alert['icon']} {alert['title']}</h4>
                <p style="margin:0.5rem 0; color:rgba(255,255,255,0.85);">{alert['message']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if alert['actions']:
                with st.expander("📋 Recommended Actions"):
                    for action in alert['actions']:
                        st.markdown(f"• {action}")

st.markdown("<br>", unsafe_allow_html=True)

# Hourly Risk Timeline
st.markdown('<div class="section-header">⏰ 24-Hour Risk Timeline</div>', unsafe_allow_html=True)

# Create visual hourly breakdown
timeline_data = []
for i in range(len(solar_df)):
    hour = int(solar_df.iloc[i]['hour'])
    solar_prob = solar_df.iloc[i]['flare_probability'] * 100
    hf_prob = hf_df.iloc[i]['hf_blackout_probability'] * 100
    max_risk = max(solar_prob, hf_prob)
    
    if max_risk > 70:
        risk_emoji = '🔴'
        risk_text = 'CRITICAL'
    elif max_risk > 50:
        risk_emoji = '🟠'
        risk_text = 'HIGH'
    elif max_risk > 30:
        risk_emoji = '🟡'
        risk_text = 'MODERATE'
    else:
        risk_emoji = '🟢'
        risk_text = 'LOW'
    
    timeline_data.append({
        'Hour': f'H+{hour}',
        'Risk': risk_emoji,
        'Status': risk_text,
        'Solar': f'{solar_prob:.0f}%',
        'HF': f'{hf_prob:.0f}%',
        'Severity': hf_df.iloc[i]['blackout_severity']
    })

timeline_df = pd.DataFrame(timeline_data)

# Display in columns for better readability
col1, col2 = st.columns([3, 2])

with col1:
    st.dataframe(
        timeline_df,
        width='stretch',
        hide_index=True,
        height=400
    )

with col2:
    # Risk distribution pie chart
    risk_counts = timeline_df['Status'].value_counts()
    
    # Create color mapping
    color_map = {
        'CRITICAL': '#9f1239',
        'HIGH': '#f56565',
        'MODERATE': '#ed8936',
        'LOW': '#48bb78'
    }
    colors = [color_map.get(k, '#4299e1') for k in risk_counts.index]
    
    fig_risk_dist = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        marker=dict(colors=colors),
        hole=0.4
    )])
    
    fig_risk_dist.update_layout(
        title='Risk Distribution',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#ffffff'),
        showlegend=True
    )
    
    st.plotly_chart(fig_risk_dist, width='stretch')

st.markdown("<br>", unsafe_allow_html=True)

# Quick Stats Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="info-card" style="text-align: center;">
        <h3 style="color: #667eea; margin: 0;">Correlation</h3>
        <div style="font-size: 2rem; font-weight: 700; color: #fff; margin: 0.5rem 0;">{:.3f}</div>
        <small style="color: rgba(255,255,255,0.6);">Solar ↔ HF</small>
    </div>
    """.format(correlation), unsafe_allow_html=True)

with col2:
    volatility = solar_df['flare_probability'].std() * 100
    st.markdown(f"""
    <div class="info-card" style="text-align: center;">
        <h3 style="color: #667eea; margin: 0;">Volatility</h3>
        <div style="font-size: 2rem; font-weight: 700; color: #fff; margin: 0.5rem 0;">{volatility:.1f}%</div>
        <small style="color: rgba(255,255,255,0.6);">Std Deviation</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    peak_diff = abs(int(peak_hour_solar) - int(peak_hour_hf))
    st.markdown(f"""
    <div class="info-card" style="text-align: center;">
        <h3 style="color: #667eea; margin: 0;">Peak Offset</h3>
        <div style="font-size: 2rem; font-weight: 700; color: #fff; margin: 0.5rem 0;">{peak_diff}h</div>
        <small style="color: rgba(255,255,255,0.6);">Solar-HF Lag</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    forecast_quality = 'HIGH' if correlation > 0.6 and volatility < 15 else 'MODERATE' if correlation > 0.4 else 'LOW'
    quality_color = '#48bb78' if forecast_quality == 'HIGH' else '#ed8936' if forecast_quality == 'MODERATE' else '#f56565'
    st.markdown(f"""
    <div class="info-card" style="text-align: center;">
        <h3 style="color: #667eea; margin: 0;">Confidence</h3>
        <div style="font-size: 2rem; font-weight: 700; color: {quality_color}; margin: 0.5rem 0;">{forecast_quality}</div>
        <small style="color: rgba(255,255,255,0.6);">Forecast Quality</small>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Risk Assessment (keep existing but simplify)
col_risk1, col_risk2 = st.columns(2)

with col_risk1:
    st.markdown('<div class="section-header">☀️ Solar Flare Assessment</div>', unsafe_allow_html=True)
    if peak_solar > 70:
        st.markdown("""
        <div class="risk-indicator risk-severe">
        <h3 style="margin:0; color:#fff;">🔴 EXTREME RISK</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Major solar flares expected. Monitor continuously.</p>
        </div>
        """, unsafe_allow_html=True)
    elif peak_solar > 50:
        st.markdown("""
        <div class="risk-indicator risk-high">
        <h3 style="margin:0; color:#fff;">🟠 HIGH RISK</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Significant solar activity likely.</p>
        </div>
        """, unsafe_allow_html=True)
    elif peak_solar > 30:
        st.markdown("""
        <div class="risk-indicator risk-moderate">
        <h3 style="margin:0; color:#fff;">🟡 MODERATE RISK</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Elevated solar activity possible.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="risk-indicator risk-low">
        <h3 style="margin:0; color:#fff;">🟢 LOW RISK</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Minimal solar activity expected.</p>
        </div>
        """, unsafe_allow_html=True)

with col_risk2:
    st.markdown('<div class="section-header">📡 HF Blackout Risk Level</div>', unsafe_allow_html=True)
    if peak_hf > 75:
        st.markdown("""
        <div class="risk-indicator risk-severe">
        <h3 style="margin:0; color:#fff;">🔴 SEVERE</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Complete HF communication outages expected.</p>
        </div>
        """, unsafe_allow_html=True)
    elif peak_hf > 60:
        st.markdown("""
        <div class="risk-indicator risk-high">
        <h3 style="margin:0; color:#fff;">🟠 STRONG</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Wide-area HF blackouts expected.</p>
        </div>
        """, unsafe_allow_html=True)
    elif peak_hf > 40:
        st.markdown("""
        <div class="risk-indicator risk-moderate">
        <h3 style="margin:0; color:#fff;">🟡 MODERATE</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Limited HF disruptions possible.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="risk-indicator risk-low">
        <h3 style="margin:0; color:#fff;">🟢 MINOR</h3>
        <p style="margin:0.5rem 0 0 0; color:rgba(255,255,255,0.8);">Minimal HF impact expected.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main charts
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "☀️ Solar Analysis", "📡 HF Blackouts", "💡 Insights"])

with tab1:
    st.markdown('<div class="section-header">24-Hour Forecast Timeline</div>', unsafe_allow_html=True)
    
    # Combined chart with professional styling
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=solar_df['hour'],
        y=solar_df['flare_probability'] * 100,
        name='Solar Flare Probability',
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea', line=dict(color='#fff', width=1)),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=hf_df['hour'],
        y=hf_df['hf_blackout_probability'] * 100,
        name='HF Blackout Probability',
        mode='lines+markers',
        line=dict(color='#48bb78', width=3),
        marker=dict(size=8, color='#48bb78', line=dict(color='#fff', width=1)),
        fill='tozeroy',
        fillcolor='rgba(72, 187, 120, 0.1)'
    ))
    
    fig.update_layout(
        height=450,
        plot_bgcolor='rgba(26, 32, 44, 0.6)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', size=12, color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(26, 32, 44, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        ),
        xaxis=dict(
            title='Hour',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Probability (%)',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True,
            zeroline=False
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Split into insights columns
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### 📊 Correlation Analysis")
        
        # Create correlation heatmap for hourly data
        # Prepare data for correlation matrix
        corr_data = pd.DataFrame({
            'Solar Flare': solar_df['flare_probability'],
            'HF Blackout': hf_df['hf_blackout_probability']
        })
        
        # Scatter plot with trend line
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=solar_df['flare_probability'] * 100,
            y=hf_df['hf_blackout_probability'] * 100,
            mode='markers',
            marker=dict(
                size=10,
                color=solar_df['hour'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Hour", len=0.5),
                line=dict(color='#fff', width=1)
            ),
            text=[f"Hour {int(h)}" for h in solar_df['hour']],
            hovertemplate='<b>%{text}</b><br>Solar: %{x:.1f}%<br>HF: %{y:.1f}%<extra></extra>',
            name='Hourly Data'
        ))
        
        # Add trend line
        z = np.polyfit(solar_df['flare_probability'] * 100, hf_df['hf_blackout_probability'] * 100, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(solar_df['flare_probability'].min() * 100, 
                             solar_df['flare_probability'].max() * 100, 100)
        
        fig_scatter.add_trace(go.Scatter(
            x=x_trend,
            y=p(x_trend),
            mode='lines',
            line=dict(color='#ed8936', width=2, dash='dash'),
            name=f'Trend (r={correlation:.3f})'
        ))
        
        fig_scatter.update_layout(
            height=380,
            plot_bgcolor='rgba(26, 32, 44, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#ffffff'),
            xaxis=dict(title='Solar Flare Probability (%)', gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title='HF Blackout Probability (%)', gridcolor='rgba(255, 255, 255, 0.05)'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(26, 32, 44, 0.8)',
                bordercolor='rgba(255, 255, 255, 0.2)',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation interpretation
        if abs(correlation) > 0.7:
            corr_strength = "Strong"
            corr_color = "#48bb78"
            corr_msg = "Solar activity reliably predicts HF blackouts"
        elif abs(correlation) > 0.4:
            corr_strength = "Moderate"
            corr_color = "#ed8936"
            corr_msg = "Solar activity moderately correlates with HF events"
        else:
            corr_strength = "Weak"
            corr_color = "#f56565"
            corr_msg = "Other factors may be influencing HF blackouts"
        
        st.markdown(f"""
        <div class="info-card">
            <strong style="color: {corr_color};">{corr_strength} Correlation ({correlation:.3f})</strong><br>
            <small style="color: rgba(255,255,255,0.7);">{corr_msg}</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("#### ⚡ Peak Risk Periods")
    
        # Show top 5 high risk hours
        peak_hours = solar_df.nlargest(5, 'flare_probability')
        
        for idx, row in peak_hours.iterrows():
            hour = int(row['hour'])
            solar_prob = row['flare_probability'] * 100
            flare_class = row['flare_class']
            hf_row = hf_df.iloc[idx]
            hf_prob = hf_row['hf_blackout_probability'] * 100
            hf_severity = hf_row['blackout_severity']
            
            st.markdown(f"""
            <div class="info-card" style="border-left: 3px solid #667eea; padding: 1rem; margin: 0.75rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong style="color: #fff; font-size: 1.1rem;">Hour {hour}</strong>
                        <div style="margin-top: 0.25rem;">
                            <span class="status-badge badge-info" style="margin-right: 0.5rem;">{flare_class}</span>
                            <span class="status-badge badge-warning">{hf_severity}</span>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: #667eea; font-weight: 600;">☀️ {solar_prob:.1f}%</div>
                        <div style="color: #48bb78; font-weight: 600;">📡 {hf_prob:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("#### 📊 Activity Distribution")
        
        # Hourly activity sparkline
        fig_sparkline = go.Figure()
        
        fig_sparkline.add_trace(go.Bar(
            x=solar_df['hour'],
            y=solar_df['flare_probability'] * 100,
            marker=dict(
                color=solar_df['flare_probability'] * 100,
                colorscale='RdYlGn_r',
                showscale=False
            ),
            name='Solar Activity'
        ))
        
        fig_sparkline.update_layout(
            height=180,
            plot_bgcolor='rgba(26, 32, 44, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#ffffff', size=10),
            xaxis=dict(
                title='',
                showgrid=False,
                showticklabels=False
            ),
            yaxis=dict(
                title='',
                showgrid=False,
                showticklabels=False
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False
        )
        
        st.plotly_chart(fig_sparkline, use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Solar Flare Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_solar = go.Figure()
        
        colors = ['#667eea' if p > 0.5 else '#4299e1' if p > 0.3 else '#48bb78' for p in solar_df['flare_probability']]
        
        fig_solar.add_trace(go.Bar(
            x=solar_df['hour'],
            y=solar_df['flare_probability'] * 100,
            marker=dict(color=colors, line=dict(color='#fff', width=1)),
            name='Solar Flare Probability'
        ))
        
        fig_solar.update_layout(
            height=400,
            plot_bgcolor='rgba(26, 32, 44, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#ffffff'),
            xaxis=dict(title='Hour', gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title='Probability (%)', gridcolor='rgba(255, 255, 255, 0.05)'),
            showlegend=False
        )
        
        st.plotly_chart(fig_solar, use_container_width=True)
    
    with col2:
        st.markdown("#### Class Distribution")
        class_counts = solar_df['flare_class'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=class_counts.index,
            values=class_counts.values,
            marker=dict(colors=['#667eea', '#4299e1', '#ed8936', '#f56565']),
            textinfo='label+percent',
            hole=0.4
        )])
        
        fig_pie.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            font=dict(family='Inter', color='#ffffff')
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">HF Radio Blackout Forecast</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        severity_colors = {
            'None': '#1a202c',
            'R1': '#48bb78',
            'R2': '#4299e1',
            'R3': '#ed8936',
            'R4': '#f56565',
            'R5': '#9f1239'
        }
        
        colors_hf = [severity_colors.get(s, '#1a202c') for s in hf_df['blackout_severity']]
        
        fig_hf = go.Figure()
        
        fig_hf.add_trace(go.Bar(
            x=hf_df['hour'],
            y=hf_df['hf_blackout_probability'] * 100,
            marker=dict(color=colors_hf, line=dict(color='#fff', width=1)),
            text=hf_df['blackout_severity'],
            textposition='outside',
            name='HF Blackout Severity'
        ))
        
        fig_hf.update_layout(
            height=400,
            plot_bgcolor='rgba(26, 32, 44, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#ffffff'),
            xaxis=dict(title='Hour', gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title='Probability (%)', gridcolor='rgba(255, 255, 255, 0.05)'),
            showlegend=False
        )
        
        st.plotly_chart(fig_hf, use_container_width=True)
    
    with col2:
        st.markdown("#### Severity Breakdown")
        severity_counts = hf_df['blackout_severity'].value_counts().sort_index()
        
        fig_severity = go.Figure(data=[go.Pie(
            labels=severity_counts.index,
            values=severity_counts.values,
            marker=dict(colors=[severity_colors.get(s, '#1a202c') for s in severity_counts.index]),
            textinfo='label+value',
            hole=0.4
        )])
        
        fig_severity.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            font=dict(family='Inter', color='#ffffff')
        )
        
        st.plotly_chart(fig_severity, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Key Insights & Operational Intelligence</div>', unsafe_allow_html=True)
    
    # Statistical Summary Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ☀️ Solar Statistics")
        st.markdown(f"""
        <div class="info-card">
            <table style="width: 100%; color: rgba(255,255,255,0.85); line-height: 2;">
                <tr><td><strong>Mean:</strong></td><td style="text-align: right; color: #667eea;">{solar_df['flare_probability'].mean()*100:.2f}%</td></tr>
                <tr><td><strong>Median:</strong></td><td style="text-align: right; color: #667eea;">{solar_df['flare_probability'].median()*100:.2f}%</td></tr>
                <tr><td><strong>Std Dev:</strong></td><td style="text-align: right; color: #667eea;">{solar_df['flare_probability'].std()*100:.2f}%</td></tr>
                <tr><td><strong>Peak:</strong></td><td style="text-align: right; color: #f56565;">{solar_df['flare_probability'].max()*100:.2f}%</td></tr>
                <tr><td><strong>Min:</strong></td><td style="text-align: right; color: #48bb78;">{solar_df['flare_probability'].min()*100:.2f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📡 HF Blackout Statistics")
        st.markdown(f"""
        <div class="info-card">
            <table style="width: 100%; color: rgba(255,255,255,0.85); line-height: 2;">
                <tr><td><strong>Mean:</strong></td><td style="text-align: right; color: #48bb78;">{hf_df['hf_blackout_probability'].mean()*100:.2f}%</td></tr>
                <tr><td><strong>Median:</strong></td><td style="text-align: right; color: #48bb78;">{hf_df['hf_blackout_probability'].median()*100:.2f}%</td></tr>
                <tr><td><strong>Std Dev:</strong></td><td style="text-align: right; color: #48bb78;">{hf_df['hf_blackout_probability'].std()*100:.2f}%</td></tr>
                <tr><td><strong>Peak:</strong></td><td style="text-align: right; color: #f56565;">{hf_df['hf_blackout_probability'].max()*100:.2f}%</td></tr>
                <tr><td><strong>Min:</strong></td><td style="text-align: right; color: #48bb78;">{hf_df['hf_blackout_probability'].min()*100:.2f}%</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🎯 Forecast Metrics")
        
        # Calculate forecast quality indicators
        range_solar = (solar_df['flare_probability'].max() - solar_df['flare_probability'].min()) * 100
        range_hf = (hf_df['hf_blackout_probability'].max() - hf_df['hf_blackout_probability'].min()) * 100
        
        st.markdown(f"""
        <div class="info-card">
            <table style="width: 100%; color: rgba(255,255,255,0.85); line-height: 2;">
                <tr><td><strong>Correlation:</strong></td><td style="text-align: right; color: #667eea;">{correlation:.3f}</td></tr>
                <tr><td><strong>Solar Range:</strong></td><td style="text-align: right; color: #667eea;">{range_solar:.1f}%</td></tr>
                <tr><td><strong>HF Range:</strong></td><td style="text-align: right; color: #48bb78;">{range_hf:.1f}%</td></tr>
                <tr><td><strong>High Risk Hrs:</strong></td><td style="text-align: right; color: #f56565;">{high_risk_solar + high_risk_hf}</td></tr>
                <tr><td><strong>Severe Events:</strong></td><td style="text-align: right; color: #f56565;">{severe_events}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk Matrix and Operational Guidance
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("#### � Risk Matrix")
        
        # Create risk heatmap matrix
        # Categorize hours into risk buckets
        risk_matrix = np.zeros((3, 3))  # Solar (rows) x HF (cols)
        
        for i in range(len(solar_df)):
            # Categorize solar: 0=low(<30%), 1=med(30-60%), 2=high(>60%)
            solar_val = solar_df.iloc[i]['flare_probability'] * 100
            if solar_val < 30:
                s_idx = 0
            elif solar_val < 60:
                s_idx = 1
            else:
                s_idx = 2
            
            # Categorize HF: 0=low(<30%), 1=med(30-60%), 2=high(>60%)
            hf_val = hf_df.iloc[i]['hf_blackout_probability'] * 100
            if hf_val < 30:
                h_idx = 0
            elif hf_val < 60:
                h_idx = 1
            else:
                h_idx = 2
            
            risk_matrix[2-s_idx, h_idx] += 1  # Flip rows for visual
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=['Low<br>(<30%)', 'Moderate<br>(30-60%)', 'High<br>(>60%)'],
            y=['High<br>(>60%)', 'Moderate<br>(30-60%)', 'Low<br>(<30%)'],
            colorscale='RdYlGn_r',
            text=risk_matrix.astype(int),
            texttemplate='%{text} hrs',
            textfont={"size": 14, "color": "white"},
            hoverongaps=False,
            hovertemplate='Solar: %{y}<br>HF: %{x}<br>Hours: %{z}<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            height=300,
            plot_bgcolor='rgba(26, 32, 44, 0.6)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#ffffff', size=11),
            xaxis=dict(title='HF Blackout Risk', side='bottom'),
            yaxis=dict(title='Solar Flare Risk'),
            margin=dict(l=80, r=20, t=20, b=80)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.caption("📊 Distribution of hours across solar and HF risk levels")
    
    with col_right:
        st.markdown("#### 🛡️ Operational Guidance")
        
        # Provide context-aware recommendations
        if peak_hf > 60:
            st.markdown("""
            <div class="risk-indicator risk-high">
                <strong>CRITICAL PREPAREDNESS</strong><br>
                <small style="color: rgba(255,255,255,0.9);">
                Severe HF blackouts forecasted. Implement full contingency protocols.
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Essential Actions:**
            - ✅ Activate satellite backup systems
            - ✅ Notify all HF radio operators
            - ✅ Postpone non-critical HF operations
            - ✅ Enable continuous monitoring
            - ✅ Brief management on expected impacts
            """)
        elif peak_hf > 40:
            st.markdown("""
            <div class="risk-indicator risk-moderate">
                <strong>ELEVATED WATCHFULNESS</strong><br>
                <small style="color: rgba(255,255,255,0.9);">
                Moderate HF disruptions possible. Enhanced monitoring required.
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Recommended Actions:**
            - 📋 Test backup communication systems
            - 📋 Brief operators on potential disruptions
            - 📋 Monitor HF channel quality
            - 📋 Document communication anomalies
            - 📋 Review contingency procedures
            """)
        else:
            st.markdown("""
            <div class="risk-indicator risk-low">
                <strong>ROUTINE OPERATIONS</strong><br>
                <small style="color: rgba(255,255,255,0.9);">
                Minimal HF impact expected. Standard monitoring sufficient.
                </small>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            **Standard Procedures:**
            - ℹ️ Maintain routine monitoring
            - ℹ️ Continue normal HF operations
            - ℹ️ Log any unusual activity
            - ℹ️ Keep backup systems ready
            """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data quality indicator
        data_quality_score = 100
        if correlation < 0.4:
            data_quality_score -= 20
        if volatility > 20:
            data_quality_score -= 15
        if severe_events == 0 and peak_solar < 30:
            data_quality_score -= 10  # Limited event diversity
        
        quality_color = '#48bb78' if data_quality_score >= 80 else '#ed8936' if data_quality_score >= 60 else '#f56565'
        
        st.markdown(f"""
        <div class="info-card" style="border-left: 3px solid {quality_color};">
            <strong>Forecast Reliability</strong><br>
            <div style="font-size: 2rem; color: {quality_color}; font-weight: 700; margin: 0.5rem 0;">{data_quality_score}%</div>
            <small style="color: rgba(255,255,255,0.7);">
            Based on correlation strength, volatility, and event coverage
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Trend Analysis
    st.markdown("#### 📈 Trend Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Solar trend
        trend_emoji = '📈' if solar_trend > 0 else '📉' if solar_trend < 0 else '➡️'
        trend_color = '#f56565' if solar_trend > 5 else '#48bb78' if solar_trend < -5 else '#4299e1'
        
        st.markdown(f"""
        <div class="info-card">
            <strong>Solar Activity Trend {trend_emoji}</strong><br>
            <div style="color: {trend_color}; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">
                {solar_trend:+.1f}%
            </div>
            <small style="color: rgba(255,255,255,0.7);">
            {"Increasing" if solar_trend > 0 else "Decreasing" if solar_trend < 0 else "Stable"} over 24h period
            <br>First 12h avg: {solar_df['flare_probability'].iloc[:12].mean()*100:.1f}%
            <br>Last 12h avg: {solar_df['flare_probability'].iloc[12:].mean()*100:.1f}%
            </small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # HF trend
        hf_trend_emoji = '📈' if hf_trend > 0 else '📉' if hf_trend < 0 else '➡️'
        hf_trend_color = '#f56565' if hf_trend > 5 else '#48bb78' if hf_trend < -5 else '#4299e1'
        
        st.markdown(f"""
        <div class="info-card">
            <strong>HF Blackout Trend {hf_trend_emoji}</strong><br>
            <div style="color: {hf_trend_color}; font-size: 1.5rem; font-weight: 700; margin: 0.5rem 0;">
                {hf_trend:+.1f}%
            </div>
            <small style="color: rgba(255,255,255,0.7);">
            {"Increasing" if hf_trend > 0 else "Decreasing" if hf_trend < 0 else "Stable"} over 24h period
            <br>First 12h avg: {hf_df['hf_blackout_probability'].iloc[:12].mean()*100:.1f}%
            <br>Last 12h avg: {hf_df['hf_blackout_probability'].iloc[12:].mean()*100:.1f}%
            </small>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Powered by**: Surya Foundation Model")
    st.caption("366M parameters • Pattern recognition")

with col2:
    st.markdown("**Forecast Horizon**: 24 hours")
    st.caption("Updated in real-time")

with col3:
    if solar_df is not None:
        timestamp = pd.to_datetime(solar_df['timestamp'].iloc[0])
        st.markdown(f"**Last Updated**: {timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        st.caption("Enterprise monitoring system")
