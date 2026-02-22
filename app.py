import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import json
import os
import textwrap
from datetime import datetime

# --- Page Config ---
st.set_page_config(
    page_title="AI Maintainance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet" />', unsafe_allow_html=True)

# --- Constants & Config ---
MODEL_FILE = 'final_model.pkl'
SCALER_FILE = 'final_scaler.pkl'

# --- Session State Initialization ---
if 'history_logs' not in st.session_state:
    # Initialize with some mock data
    initial_data = []
    for i in range(10):
        is_fail = np.random.choice([0, 1], p=[0.8, 0.2])
        initial_data.append({
            "Timestamp": (datetime.now() - pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "Machine ID": f"MACH-{np.random.choice(['X99', 'A02', 'B15', 'C33', 'D41'])}",
            "Machine Type": np.random.choice(['Type M', 'Type H', 'Type L']),
            "Temperature": np.random.randint(40, 95),
            "Predicted Failure": bool(is_fail),
            "Probability": np.random.randint(85, 99) if is_fail else np.random.randint(5, 40),
            "Engineer Validation": "Pending",
            "Prediction Accuracy": "Unknown"
        })
    st.session_state['history_logs'] = initial_data

if 'last_prediction' not in st.session_state:
    st.session_state['last_prediction'] = None

# --- Theme & Styling ---
STYLING_CSS_GLOBAL = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');
    
    /* GLOBAL THEME - Teal Industrial */
    .stApp {
        background-color: #0F172A;
        background-image: 
            linear-gradient(rgba(20, 184, 166, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 184, 166, 0.04) 1px, transparent 1px);
        background-size: 50px 50px;
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
        border-right: 1px solid rgba(20, 184, 166, 0.15);
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] div {
        color: #F8FAFC !important;
    }
    
    /* Nav radio - pill style */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 4px;
        padding: 4px;
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.2s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(20, 184, 166, 0.1);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background: rgba(20, 184, 166, 0.2);
        color: #14B8A6 !important;
    }

    /* WIDGET STYLING */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: rgba(20, 184, 166, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.15) !important;
    }

    /* Fix Number Input Buttons */
    [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {
        color: #F8FAFC !important;
        background-color: #334155 !important;
        border-left: 1px solid #475569 !important;
        border-radius: 0 8px 8px 0 !important;
        opacity: 1 !important;
    }

    /* GLASS CARDS */
    .glass-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(20, 184, 166, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }

    /* Primary button styling */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(20, 184, 166, 0.4) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(20, 184, 166, 0.25) !important;
        transform: translateY(-1px);
    }
    
    /* Subheaders */
    h2, h3 {
        letter-spacing: -0.02em !important;
        font-weight: 600 !important;
    }
    
    /* Dividers */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(20, 184, 166, 0.25), transparent);
        margin: 1.5rem 0 !important;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* HIDE DEFAULT STREAMLIT ELEMENTS */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""

st.markdown(STYLING_CSS_GLOBAL, unsafe_allow_html=True)

# --- Resource Loading ---
@st.cache_resource
def load_resources():
    try:
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler
    except Exception:
        pass
    return None, None

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 1.25rem 0; text-align: center; border-bottom: 1px solid rgba(20, 184, 166, 0.15); margin-bottom: 1rem;">
        <div style="width: 56px; height: 56px; margin: 0 auto 12px; background: linear-gradient(135deg, rgba(20, 184, 166, 0.25), rgba(20, 184, 166, 0.06)); border-radius: 14px; display: flex; align-items: center; justify-content: center;">
            <span class="material-symbols-outlined" style="font-size: 32px; color: #14B8A6;">precision_manufacturing</span>
        </div>
        <h3 style="margin:0; font-size: 1.25rem; font-weight: 700; letter-spacing: -0.03em;">AI Maintainance</h3>
        <p style="font-size: 0.75rem; color: #A8A29E !important; margin-top: 6px;">
            <span style="display: inline-flex; align-items: center; gap: 4px;"><span style="width: 6px; height: 6px; background: #22C55E; border-radius: 50%; animation: pulse 2s infinite;"></span> Online</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio("MENU", ["📊 System Dashboard", "🧠 Prediction Engine", "📜 Alert Logs"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown(f"<div style='padding: 12px 14px; background: rgba(0,0,0,0.3); border-radius: 10px; border: 1px solid rgba(20, 184, 166, 0.15); font-size: 0.85rem;'><span style='color: #A8A29E;'>System Operator</span><br><b style='color: #F8FAFC;'>Standard Session</b></div>", unsafe_allow_html=True)

# --- Main Content Area ---

# 1. DASHBOARD
if "Dashboard" in page:
    c_head1, c_head2, c_head3 = st.columns([2, 4, 1])
    with c_head1:
        st.markdown("""
        <div style='color: #A8A29E; font-size: 0.8rem; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;'>
            <span style='color: #64748B;'>Operations</span>
            <span style='color: #64748B;'>›</span>
            <span>Unit A4 Monitor</span>
            <span style='background: linear-gradient(135deg, #14B8A6, #0D9488); color:#0F172A; padding: 3px 8px; border-radius: 6px; font-weight: 700; font-size: 0.65rem; letter-spacing: 0.5px;'>ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
        st.title("System Dashboard")
    with c_head3:
        st.markdown("<div style='text-align: right; padding-top: 20px;'><span class='material-symbols-outlined' style='color: #A8A29E;'>notifications</span> <span class='material-symbols-outlined' style='color: #A8A29E; margin-left: 10px;'>help</span></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <span style='color: #A8A29E; font-size: 0.85rem; font-weight: 500;'>Key metrics</span>
    </div>
    """, unsafe_allow_html=True)
    kpi1, kpi2, kpi3 = st.columns(3)
    
    with kpi1:
        st.markdown("""
        <div class="glass-card" style="height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #A8A29E; font-size: 0.9rem; margin-bottom: 10px;">● Current Failure Risk</div>
                <div style="font-size: 3.5rem; font-weight: 700; color: #F8FAFC;">12<span style="font-size: 1.5rem; color: #A8A29E;">%</span></div>
            </div>
            <div>
                <div style="background: rgba(34, 197, 94, 0.25); color: #22C55E; padding: 4px 8px; border-radius: 4px; display: inline-block; font-size: 0.8rem; font-weight: 600;">↘ -5% from avg</div>
                <div style="height: 4px; background: #334155; border-radius: 2px; margin-top: 10px; width: 100%;">
                    <div style="height: 100%; width: 12%; background: #14B8A6; border-radius: 2px;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        import plotly.graph_objects as go
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = 88,
            title = {'text': "System Health Status", 'font': {'size': 14, 'color': "#A8A29E"}},
            number = {'suffix': "/100", 'font': {'size': 24, 'color': "#F8FAFC"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#334155"},
                'bar': {'color': "#14B8A6"},
                'bgcolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.25)"},
                    {'range': [60, 85], 'color': "rgba(20, 184, 166, 0.2)"},
                    {'range': [85, 100], 'color': "rgba(34, 197, 94, 0.25)"}
                ],
                'threshold': {'line': {'color': "#14B8A6", 'width': 2}, 'thickness': 0.75, 'value': 88}
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(30, 41, 59, 0.5)', font={'color': "#F8FAFC"}, height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.markdown('<div class="glass-card" style="padding: 0;">', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi3:
        st.markdown("""
        <div class="glass-card" style="height: 280px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="color: #A8A29E; font-size: 0.9rem; margin-bottom: 10px;">✨ Predicted Status</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: #F8FAFC; letter-spacing: 1px;">● NORMAL</div>
            </div>
            <div>
                <div style="display: flex; justify-content: space-between; color: #A8A29E; font-size: 0.8rem; margin-bottom: 5px;">
                    <span>Confidence Level</span>
                    <span style="color: #F8FAFC; font-weight: 600;">94%</span>
                </div>
                <div style="height: 6px; background: #334155; border-radius: 3px; width: 100%;">
                    <div style="height: 100%; width: 94%; background: #14B8A6; border-radius: 3px;"></div>
                </div>
                <div style="margin-top: 15px; font-size: 0.75rem; color: #64748B;">Next predicted maintenance: <br><span style="color: #A8A29E;">14 Days</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Performance Analytics")
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Vibration Analysis")
        x = np.linspace(0, 24, 100)
        y = 30 + 10 * np.sin(x/2) + np.random.normal(0, 2, 100)
        df_vib = pd.DataFrame({'Time': x, 'Vibration': y})
        import plotly.express as px
        fig_vib = px.area(df_vib, x='Time', y='Vibration', template='plotly_dark')
        fig_vib.update_traces(line_color='#14B8A6', fillcolor='rgba(20, 184, 166, 0.15)')
        fig_vib.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#F8FAFC'}, height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_vib, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_chart2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Temperature Loads")
        df_temp = pd.DataFrame({'Unit': ['Unit 1', 'Unit 2', 'Unit 3', 'Unit 4', 'Unit 5'], 'Temp': [65, 45, 55, 80, 60]})
        fig_temp = px.bar(df_temp, x='Unit', y='Temp', template='plotly_dark')
        colors = ['#334155'] * 5
        colors[3] = '#14B8A6'
        fig_temp.update_traces(marker_color=colors)
        fig_temp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': '#F8FAFC'}, height=220, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_temp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 2. PREDICTION ENGINE
elif "Prediction" in page:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;'>
        <span class='material-symbols-outlined' style='font-size: 32px; color: #14B8A6;'>psychology</span>
        <div>
            <h1 style='margin: 0; font-size: 1.75rem; font-weight: 700;'>Neural Prediction Engine</h1>
            <p style='margin: 4px 0 0 0; color: #A8A29E; font-size: 0.9rem;'>Run failure predictions from sensor data</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    model, scaler = load_resources()
    
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
            <span class='material-symbols-outlined' style='font-size: 22px; color: #14B8A6;'>sensors</span>
            <h3 style='margin: 0; font-size: 1.1rem;'>Sensor Inputs</h3>
        </div>
        """, unsafe_allow_html=True)
        machine_id = st.text_input("Machine ID", value="MACH-PRX1")
        machine_type = st.selectbox("Machine Type", ["Type M", "Type H", "Type L"], help="Select the machine type for accurate predictions")
        temp_input = st.number_input("Temperature [K]", 290.0, 310.0, 300.0)
        rpm_input = st.number_input("RPM (Rotational Speed)", 1000, 3000, 1500)
        tool_wear_input = st.number_input("Tool Wear [min]", 0, 300, 0)
        pressure_input = st.number_input("Pressure [Bar] (Mapped to Torque)", 0.0, 100.0, 40.0)
        vib_input = st.number_input("Vibration [Hz]", 0.0, 100.0, 50.0)

        # Dynamically set one-hot encoding based on selected machine type
        type_h = 1 if machine_type == "Type H" else 0
        type_l = 1 if machine_type == "Type L" else 0
        type_m = 1 if machine_type == "Type M" else 0

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔬 ANALYZE SENSOR DATA", use_container_width=True, type="primary"):
            if model and scaler:
                input_df = pd.DataFrame([{
                    'UDI': 0, 'Air_temperature_K': temp_input, 'Process_temperature_K': temp_input + 10,
                    'Rotational_speed_rpm': rpm_input, 'Torque_Nm': pressure_input, 'Tool_wear_min': tool_wear_input,
                    'Type_H': type_h, 'Type_L': type_l, 'Type_M': type_m
                }])
                try:
                    scaled = scaler.transform(input_df)
                    prob = model.predict_proba(scaled)[0][1]
                    st.session_state['last_prediction'] = {
                        "Machine ID": machine_id,
                        "Machine Type": machine_type,
                        "Failure Predicted": prob > 0.5,
                        "Probability": prob,
                        "Temperature": temp_input
                    }
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
            else:
                # Simulation Mode
                prob = np.random.random()
                st.session_state['last_prediction'] = {
                    "Machine ID": machine_id,
                    "Machine Type": machine_type,
                    "Failure Predicted": prob > 0.5,
                    "Probability": prob,
                    "Temperature": temp_input
                }
        st.markdown("</div>", unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
            <span class='material-symbols-outlined' style='font-size: 22px; color: #14B8A6;'>analytics</span>
            <h3 style='margin: 0; font-size: 1.1rem;'>Analysis Results</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state['last_prediction']:
            res = st.session_state['last_prediction']
            prob = res['Probability']
            machine_type_display = res.get('Machine Type', 'N/A')
            st.write(f"**Machine Type:** {machine_type_display}")
            st.write(f"**Failure Probability: {prob:.1%}**")
            st.progress(int(prob * 100))
            
            if res['Failure Predicted']:
                st.markdown('<div style="background: rgba(239, 68, 68, 0.15); border-left: 4px solid #EF4444; padding: 20px; border-radius: 12px; border: 1px solid rgba(239, 68, 68, 0.2);"><div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;"><span class="material-symbols-outlined" style="color: #EF4444; font-size: 24px;">warning</span><h2 style="color: #EF4444 !important; margin:0;">CRITICAL RISK</h2></div><p style="margin: 0; color: #A8A29E;">Model predicts failure likely.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="background: rgba(34, 197, 94, 0.15); border-left: 4px solid #22C55E; padding: 20px; border-radius: 12px; border: 1px solid rgba(34, 197, 94, 0.2);"><div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;"><span class="material-symbols-outlined" style="color: #22C55E; font-size: 24px;">check_circle</span><h2 style="color: #22C55E !important; margin:0;">SYSTEM STABLE</h2></div><p style="margin: 0; color: #A8A29E;">Normal operations predicted.</p></div>', unsafe_allow_html=True)
            
            # --- Engineer Validation Section ---
            st.markdown("---")
            st.subheader("👨‍🔧 Engineer Validation")
            st.info("Please confirm if the model's prediction matches the real-world state of the machine.")
            
            v_col1, v_col2 = st.columns(2)
            
            def save_validation(is_correct):
                new_entry = {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Machine ID": res['Machine ID'],
                    "Machine Type": res.get('Machine Type', 'N/A'),
                    "Temperature": res['Temperature'],
                    "Predicted Failure": res['Failure Predicted'],
                    "Probability": int(res['Probability'] * 100),
                    "Engineer Validation": "Verified" if is_correct else "Incorrect",
                    "Prediction Accuracy": "True Prediction" if is_correct else "False Prediction"
                }
                st.session_state['history_logs'].insert(0, new_entry)
                st.session_state['last_prediction'] = None
                st.success("Validation logged successfully!")
                time.sleep(1)
                st.rerun()

            with v_col1:
                if st.button("✅ Prediction is Correct", use_container_width=True):
                    save_validation(True)
            
            with v_col2:
                if st.button("❌ Prediction is False", use_container_width=True):
                    save_validation(False)
        else:
            st.info("Input sensor data and run analysis to see results and provide validation.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# 3. ALERT LOGS
elif "Logs" in page:
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 1rem;'>
        <span class='material-symbols-outlined' style='font-size: 32px; color: #14B8A6;'>history</span>
        <div>
            <h1 style='margin: 0; font-size: 1.75rem; font-weight: 700;'>Alert History Log</h1>
            <p style='margin: 4px 0 0 0; color: #A8A29E; font-size: 0.9rem;'>View and export validation records</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    c_filter, c_export = st.columns([6, 2])
    with c_filter:
        st.text_input("Search Logs...", placeholder="Search logs...", label_visibility="collapsed")
    
    df_logs = pd.DataFrame(st.session_state['history_logs'])
    
    with c_export:
        st.download_button("📥 Export Report", df_logs.to_csv(index=False).encode('utf-8'), "alerts.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="glass-card" style="padding: 0; overflow: hidden;">', unsafe_allow_html=True)
    # Custom display with validation status
    st.dataframe(
        df_logs,
        column_config={
            "Predicted Failure": st.column_config.CheckboxColumn("Predicted Failure"),
            "Probability": st.column_config.ProgressColumn("Confidence", format="%d%%", min_value=0, max_value=100),
            "Prediction Accuracy": st.column_config.TextColumn("Accuracy Label")
        },
        use_container_width=True, 
        height=600,
        hide_index=True
    )
    st.markdown('</div>', unsafe_allow_html=True)