import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from drift_detector import DriftDetector
from model_pipeline import train_and_save_model

# --- PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Spectre MLOps Platform", page_icon="📡", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #09090b; color: #10b981; }
    .stButton>button { background-color: #10b981; color: black; font-weight: bold; width: 100%; border-radius: 6px;}
    .log-box { font-family: monospace; background-color: #111827; padding: 15px; border-radius: 8px; height: 250px; overflow-y: auto; border: 1px solid #1f2937; color: #e5e7eb;}
    .story-box { background-color: #1e1b4b; padding: 15px; border-radius: 8px; border-left: 5px solid #6366f1; margin-bottom: 20px;}
    </style>
    """, unsafe_allow_html=True)

# --- DYNAMIC SCENARIO CONFIGURATION ---
SCENARIOS = {
    "🚕 Ride-Share ETAs": {
        "f1_name": "Distance (km)", "f2_name": "Duration (mins)",
        "base_f1": (10.0, 3.0), "base_f2": (20.0, 5.0),
        "anomaly_f1": (10.0, 3.0), "anomaly_f2": (45.0, 10.0),
        "story": "<b>Baseline:</b> AI predicts Sunny Day traffic. <br><b>Drift Anomaly:</b> A Monsoon Storm floods the city, doubling commute times. The AI must be retrained to issue accurate ETAs."
    },
    "💳 Financial Fraud Prevention": {
        "f1_name": "Transaction Amt ($)", "f2_name": "Distance from Home (km)",
        "base_f1": (45.0, 15.0), "base_f2": (5.0, 2.0),
        "anomaly_f1": (500.0, 100.0), "anomaly_f2": (2000.0, 500.0),
        "story": "<b>Baseline:</b> Normal user buys groceries locally. <br><b>Drift Anomaly:</b> User's card is stolen and used overseas for massive electronics purchases. AI flags the distribution shift instantly."
    },
    "🏭 IoT Manufacturing Quality": {
        "f1_name": "Vibration (Hz)", "f2_name": "Core Temp (°C)",
        "base_f1": (50.0, 2.0), "base_f2": (120.0, 5.0),
        "anomaly_f1": (50.0, 2.0), "anomaly_f2": (180.0, 15.0),
        "story": "<b>Baseline:</b> CNC Machine operates at healthy temps. <br><b>Drift Anomaly:</b> Coolant leak causes temperature spikes. AI detects the anomaly and triggers an emergency shut-off."
    }
}

# --- SESSION STATE INITIALIZATION ---
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'drift_status' not in st.session_state:
    st.session_state.drift_status = "Awaiting Traffic..."
if 'p_values' not in st.session_state:
    st.session_state.p_values = {"F1": "N/A", "F2": "N/A"}
if 'last_scenario' not in st.session_state:
    st.session_state.last_scenario = "🚕 Ride-Share ETAs"

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {message}")

def plot_distributions(baseline, incoming, f2_name):
    """Generates dynamic chart based on selected scenario."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    
    ax.hist(baseline[:, 1], bins=30, alpha=0.6, color='#3b82f6', label='Baseline Data')
    if incoming is not None:
        color = '#ef4444' if "CRITICAL" in st.session_state.drift_status else '#10b981'
        label = 'Anomaly Data' if "CRITICAL" in st.session_state.drift_status else 'Live Production Data'
        ax.hist(incoming[:, 1], bins=30, alpha=0.7, color=color, label=label)
        
    ax.set_title(f"Statistical Distribution Comparison: {f2_name}", color="white")
    ax.set_xlabel(f2_name, color="gray")
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1f2937', edgecolor='none', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#374151')
    return fig

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("📡 MLOps Platform")
    
    # 1. SCENARIO SELECTOR
    selected_scenario = st.selectbox("1️⃣ Select Industry Scenario:", list(SCENARIOS.keys()))
    cfg = SCENARIOS[selected_scenario]
    
    # Reset system if user changes scenario
    if selected_scenario != st.session_state.last_scenario:
        st.session_state.system_initialized = False
        st.session_state.current_data = None
        st.session_state.drift_status = "Awaiting Traffic..."
        st.session_state.p_values = {"F1": "N/A", "F2": "N/A"}
        st.session_state.last_scenario = selected_scenario
        st.rerun()

    st.write("---")
    
    # 2. DEPLOY MODEL
    if st.button(f"🚀 2. Deploy AI for {selected_scenario.split(' ')[0]}"):
        with st.spinner("Training Mathematical Baseline..."):
            train_and_save_model(*cfg["base_f1"], *cfg["base_f2"])
            st.session_state.system_initialized = True
            st.session_state.drift_status = "System Healthy"
            st.session_state.current_data = None
            add_log(f"[{selected_scenario}] Baseline AI deployed to production.")
        st.success("Model Deployed.")

    st.write("---")
    
    # 3. TRAFFIC SIMULATOR
    st.subheader("3️⃣ Traffic Simulator")
    if st.button("🟢 Simulate Normal Traffic"):
        if not st.session_state.system_initialized:
            st.error("Deploy the initial model first!")
        else:
            add_log("Receiving normal production traffic...")
            f1 = np.random.normal(loc=cfg["base_f1"][0], scale=cfg["base_f1"][1], size=(500, 1))
            f2 = np.random.normal(loc=cfg["base_f2"][0], scale=cfg["base_f2"][1], size=(500, 1))
            st.session_state.current_data = np.hstack((f1, f2))
            
            detector = DriftDetector()
            is_drifting, report = detector.check_drift(st.session_state.current_data)
            
            st.session_state.p_values["F1"] = f"{report['Feature_0']['p_value']:.4f}"
            st.session_state.p_values["F2"] = f"{report['Feature_1']['p_value']:.4f}"
            st.session_state.drift_status = "HEALTHY"

    if st.button("🔴 Inject Drift Anomaly"):
        if not st.session_state.system_initialized:
            st.error("Deploy the initial model first!")
        else:
            add_log("WARNING: Outlier data influx detected!")
            f1 = np.random.normal(loc=cfg["anomaly_f1"][0], scale=cfg["anomaly_f1"][1], size=(500, 1))
            f2 = np.random.normal(loc=cfg["anomaly_f2"][0], scale=cfg["anomaly_f2"][1], size=(500, 1))
            st.session_state.current_data = np.hstack((f1, f2))
            
            detector = DriftDetector()
            is_drifting, report = detector.check_drift(st.session_state.current_data)
            
            st.session_state.p_values["F1"] = f"{report['Feature_0']['p_value']:.2e}"
            st.session_state.p_values["F2"] = f"{report['Feature_1']['p_value']:.2e}"
            
            if is_drifting:
                st.session_state.drift_status = "CRITICAL DRIFT DETECTED"
                add_log("🚨 K-S Test failed! AI predictions are no longer reliable.")
                add_log("⚙️ Orchestrator triggered automated self-healing...")
                
                with st.spinner("Self-Healing: Retraining AI on new data distribution..."):
                    time.sleep(2.0)
                    np.save("baseline_X.npy", st.session_state.current_data) 
                    add_log("✅ New AI deployed. System adapted to new environment.")
                    st.session_state.drift_status = "HEALTHY (Self-Healed)"

# --- MAIN DASHBOARD ---
cfg = SCENARIOS[st.session_state.last_scenario]

st.title("📡 Spectre MLOps: Generalized AI Observability")
st.markdown(f'<div class="story-box">{cfg["story"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"📊 Live Data Distribution")
    if st.session_state.system_initialized:
        try:
            baseline = np.load("baseline_X.npy")
            fig = plot_distributions(baseline, st.session_state.current_data, cfg["f2_name"])
            st.pyplot(fig)
        except FileNotFoundError:
            st.info("Awaiting initial AI deployment...")
    else:
        st.info("👈 Initialize the environment in the sidebar.")

with col2:
    st.subheader("🛡️ AI Immune System")
    
    if "HEALTHY" in st.session_state.drift_status:
        st.success(f"**Status:** {st.session_state.drift_status}")
    elif "CRITICAL" in st.session_state.drift_status:
        st.error(f"**Status:** {st.session_state.drift_status}")
    else:
        st.warning(f"**Status:** {st.session_state.drift_status}")
        
    st.write("---")
    st.write("**Kolmogorov-Smirnov P-Values:**")
    st.caption("Threshold < 0.05 indicates mathematical divergence.")
    
    st.write(f"🔹 **{cfg['f1_name']}:** `{st.session_state.p_values['F1']}`")
    st.write(f"🔹 **{cfg['f2_name']}:** `{st.session_state.p_values['F2']}`")

st.write("---")
st.subheader("🖥️ MLOps Action Log")
log_html = "<br>".join(st.session_state.logs[-8:]) 
st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)