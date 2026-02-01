import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import time

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load Models
@st.cache_resource
def load_resources():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "sensor_model_xgb.pkl"))
        features = joblib.load(os.path.join(MODEL_DIR, "sensor_features.pkl"))
        encoder = joblib.load(os.path.join(MODEL_DIR, "machine_encoder.pkl"))
        return model, features, encoder
    except:
        return None, None, None

model, feature_names, le_machine = load_resources()

st.set_page_config(page_title="TE Digital Twin", page_icon="🏭", layout="wide")

# --- LOAD DATA ---
data_path = os.path.join(DATA_DIR, "training_data_full.csv")
if os.path.exists(data_path):
    full_data = pd.read_csv(data_path)
    full_data['date'] = pd.to_datetime(full_data['date'])
else:
    st.error("Data not found! Run script 0 and 4 first.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/TE_Connectivity_logo.svg/1200px-TE_Connectivity_logo.svg.png", width=150)
st.sidebar.title("🏭 TE Production AI")
mode = st.sidebar.radio("Select System Mode:", ["📅 Planning Mode", "🔴 Live Monitor"])

st.sidebar.markdown("---")

# =========================================================
#  MODE 1: PLANNING (Full Filters)
# =========================================================
if mode == "📅 Planning Mode":
    st.sidebar.markdown("### 🎛️ Shift Planner")
    
    # 1. Machine
    machines = sorted(full_data['machine_id'].unique())
    selected_machine = st.sidebar.selectbox("Select Machine:", machines)
    
    # 2. Tool (Context for Plan)
    machine_data = full_data[full_data['machine_id'] == selected_machine]
    if len(machine_data) > 0:
        tools = sorted(machine_data['tool_id'].unique())
        selected_tool = st.sidebar.selectbox("Select Tool / Mold:", tools)
        
        # 3. Part
        tool_data = machine_data[machine_data['tool_id'] == selected_tool]
        parts = sorted(tool_data['part_number'].unique())
        selected_part = st.sidebar.selectbox("Select Part Number:", parts)
        
        # Main UI
        st.title("📅 Planning Mode: Baseline Forecast")
        st.markdown(f"**Context:** Predicting performance for **{selected_tool}** producing **{selected_part}**.")
        
        job_data = tool_data[tool_data['part_number'] == selected_part]
        
        if len(job_data) > 0:
            # Stats
            avg_scrap_24h = job_data['scrap_quantity'].mean()
            avg_scrap_12h = avg_scrap_24h / 2
            max_scrap = job_data['scrap_quantity'].max()
            
            # --- NEW LAYOUT: 12h vs 24h ---
            st.markdown("### 🔮 Predicted Scrap Output")
            col1, col2, col3 = st.columns(3)
            
            # 12 Hours
            col1.metric("Next 12 Hours (Shift)", f"{int(avg_scrap_12h):,}", help="Projected scrap for a standard 12h shift")
            
            # 24 Hours
            col2.metric("Next 24 Hours (Full Day)", f"{int(avg_scrap_24h):,}", help="Projected scrap for a full 24h run")
            
            # Risk
            col3.metric("Worst Case Risk (24h)", f"{int(max_scrap):,}", delta="Risk Limit", delta_color="inverse")
            
            st.markdown("---")
            st.markdown("### 📈 Historical Performance Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=job_data['date'], y=job_data['scrap_quantity'], mode='lines+markers', name='History'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No historical data available for this specific job combination.")
    else:
        st.warning("No data for this machine.")

# =========================================================
#  MODE 2: MONITORING (Machine Only)
# =========================================================
elif mode == "🔴 Live Monitor":
    st.sidebar.markdown("### 📡 Live Feed Selector")
    
    # ONLY MACHINE SELECTOR
    machines = sorted(full_data['machine_id'].unique())
    selected_machine = st.sidebar.selectbox("Select Machine to Monitor:", machines)
    
    st.title("🔴 Monitoring Mode: Real-Time Digital Twin")
    st.markdown(f"**Status:** Connecting to **{selected_machine}** sensor stream...")
    
    # Get stream data
    stream_data = full_data[full_data['machine_id'] == selected_machine].sort_values(by='date')
    
    if len(stream_data) > 0:
        if st.button("▶ START LIVE FEED", type="primary"):
            
            # Layout Placeholders
            st.markdown("### ⏱️ Live Forecast & Telemetry")
            
            # Row 1: Time and Sensor
            c1, c2 = st.columns(2)
            kpi_time = c1.empty()
            kpi_sensor = c2.empty()
            
            st.markdown("---")
            
            # Row 2: The Predictions (12h & 24h)
            st.markdown("### 🔮 Real-Time Scrap Prediction")
            c3, c4 = st.columns(2)
            kpi_12h = c3.empty()
            kpi_24h = c4.empty()
            
            chart_place = st.empty()
            alert_place = st.empty()
            
            history_scrap = []
            history_dates = []
            
            # SIMULATION LOOP
            for i in range(len(stream_data)):
                row = stream_data.iloc[i]
                
                # Input
                input_data = {}
                try:
                    input_data['machine_code'] = le_machine.transform([row['machine_id']])[0]
                except:
                    input_data['machine_code'] = 0 
                
                for col in feature_names:
                    if col != 'machine_code':
                        input_data[col] = row.get(col, 0)
                
                # Predict (This is the 24h Rate)
                input_df = pd.DataFrame([input_data])
                pred_24h = model.predict(input_df)[0]
                pred_12h = pred_24h / 2  # Proportional forecast
                
                # Update UI
                history_scrap.append(pred_24h)
                history_dates.append(row['date'])
                
                # Row 1 Updates
                kpi_time.metric("⏱ Simulation Date", str(row['date'].date()))
                
                sensor_cols = [c for c in feature_names if c != 'machine_code']
                if sensor_cols:
                    val = row[sensor_cols[0]]
                    kpi_sensor.metric(f"📡 Sensor: {sensor_cols[0]}", f"{val:.1f}")
                
                # Row 2 Updates (The Predictions)
                kpi_12h.metric("Next 12 Hours", f"{int(pred_12h):,}")
                kpi_24h.metric("Next 24 Hours", f"{int(pred_24h):,}")
                
                # Alert (Based on 24h Rate)
                if pred_24h > 2000: 
                    alert_place.error(f"⚠️ HIGH RISK: Daily trend ({int(pred_24h)}) exceeds threshold!")
                else:
                    alert_place.success("✅ SYSTEM STATUS: OPTIMAL")

                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=history_dates, y=history_scrap, mode='lines', name='Live 24h Rate', line=dict(color='#ff4b4b')))
                fig.update_layout(title="Real-Time Deviation (24h Rate)", xaxis_title="Time", yaxis_title="Predicted Scrap")
                chart_place.plotly_chart(fig, use_container_width=True)
                
                time.sleep(0.5)
            
            st.success("Shift Ended.")
    else:
        st.error("No stream available for this machine.")