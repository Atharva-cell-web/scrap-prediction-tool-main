import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go

# --- CONFIG ---
# Paths are relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # .../forecasting_v2
DATA_PATH = os.path.join(BASE_DIR, "data", "daily_scrap_forecast.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "forecast_xgb.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoders.pkl")

# Page Config
st.set_page_config(page_title="TE Scrap Forecaster", page_icon="🔮", layout="wide")

# --- HEADER ---
st.title("🔮 TE Connectivity: Scrap Forecasting AI")
st.markdown("### `Level 2: Predictive Maintenance & Planning`")
st.markdown("---")

# --- LOAD DATA & MODELS ---
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data not found at {DATA_PATH}")
        return None
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_brain():
    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Run training script first.")
        return None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODER_PATH)
    return model, encoders

df = load_data()
model, encoders = load_brain()

if df is not None and model is not None:
    
    # --- SIDEBAR: USER INPUTS ---
    st.sidebar.header("🛠️ Production Planner")
    st.sidebar.markdown("Configure the shift parameters below:")
    
    # 1. Select Machine
    unique_machines = df['machine_id'].unique()
    selected_machine = st.sidebar.selectbox("Select Machine:", unique_machines)
    
    # 2. Select Tool (Filter based on machine)
    # Only show tools that have actually run on this machine
    filtered_tools = df[df['machine_id'] == selected_machine]['tool_id'].unique()
    selected_tool = st.sidebar.selectbox("Select Tool / Mold:", filtered_tools)
    
    # 3. Select Part (Filter based on Tool)
    filtered_parts = df[
        (df['machine_id'] == selected_machine) & 
        (df['tool_id'] == selected_tool)
    ]['part_number'].unique()
    
    # If no parts found (rare edge case)
    if len(filtered_parts) == 0:
        st.error("No parts found for this Machine + Tool combination.")
        st.stop()
        
    selected_part = st.sidebar.selectbox("Select Part Number:", filtered_parts)

    # --- MAIN PANEL: PREDICTION LOGIC ---
    
    # Get History for this combination
    subset = df[
        (df['machine_id'] == selected_machine) & 
        (df['tool_id'] == selected_tool) & 
        (df['part_number'] == selected_part)
    ].sort_values(by='date')

    if len(subset) < 3:
        st.warning("⚠️ Not enough historical data to make a confident prediction.")
        st.write("Need at least 3 days of history. Found:", len(subset))
    else:
        # Prepare inputs for AI
        last_row = subset.iloc[-1]
        scrap_yesterday = last_row['scrap_quantity']
        scrap_2_days_ago = subset.iloc[-2]['scrap_quantity']
        avg_last_3 = subset.iloc[-3:]['scrap_quantity'].mean()
        
        # Encode
        try:
            m_code = encoders['machine'].transform([selected_machine])[0]
            t_code = encoders['tool'].transform([selected_tool])[0]
            p_code = encoders['part'].transform([selected_part])[0]
            
            # Predict
            features = pd.DataFrame([[
                m_code, t_code, p_code, 
                scrap_yesterday, scrap_2_days_ago, avg_last_3
            ]], columns=['machine_code', 'tool_code', 'part_code', 'scrap_lag_1', 'scrap_lag_2', 'rolling_avg_3'])
            
            prediction = model.predict(features)[0]
            
            # --- DISPLAY RESULTS ---
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="📉 Predicted Scrap (Tomorrow)", value=f"{int(prediction):,}")
            
            with col2:
                st.metric(label="🔙 Yesterday's Scrap", value=f"{int(scrap_yesterday):,}", delta=f"{int(prediction - scrap_yesterday)}")
            
            with col3:
                # Risk Logic
                risk_threshold = 1000 # Adjust this based on what TE says is "Bad"
                if prediction > risk_threshold:
                    st.error("⚠️ HIGH RISK: MAINTENANCE REQUIRED")
                else:
                    st.success("✅ LOW RISK: PRODUCTION SAFE")

            # --- PLOT THE TREND ---
            st.markdown("### 📊 Trend Analysis")
            fig = go.Figure()
            
            # Historical Line
            fig.add_trace(go.Scatter(
                x=subset['date'], 
                y=subset['scrap_quantity'], 
                mode='lines+markers',
                name='Historical Scrap',
                line=dict(color='gray')
            ))
            
            # Prediction Point (Future)
            # Add 1 day to the last date
            future_date = subset['date'].iloc[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[future_date],
                y=[prediction],
                mode='markers',
                marker=dict(size=15, color='red'),
                name='AI Prediction'
            ))
            
            fig.update_layout(title=f"Scrap Trend for Tool {selected_tool}", xaxis_title="Date", yaxis_title="Scrap Quantity")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("Please ensure data and models are generated correctly.")