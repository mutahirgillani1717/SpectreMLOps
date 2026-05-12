# 📡 Spectre MLOps: Generalized AI Observability Platform

**Spectre** is a production-grade Machine Learning Operations (MLOps) platform featuring an interactive Streamlit dashboard. It is designed to monitor live AI models, detect statistical data drift in real-time, and trigger automated self-healing (retraining) loops to maintain predictive accuracy in volatile environments.

---

## 🧠 Core Architecture

* **The Predictor (`model_pipeline.py`):** Handles the dynamic training and deployment of the base model (Random Forest) while securing the "Baseline Distribution" as a mathematical point of reference.
* **The Observer (`drift_detector.py`):** Acts as the system's immune system. It continuously analyzes incoming production data streams using the **Two-Sample Kolmogorov-Smirnov (K-S) Statistical Test**.
* **The Orchestrator (`app.py`):** An event-driven Streamlit GUI that routes traffic, visualizes data distributions, catches `[CRITICAL ALERT]` drift notifications (where $p < 0.05$), and dynamically triggers the CI/CD retraining loop to redeploy a fresh model.

---

## 🎛️ Dynamic Industry Scenarios

Unlike static AI demos, Spectre features a generalized architecture capable of adapting to multiple enterprise use cases instantly via the UI:

1. **🚕 Ride-Share ETAs:** Monitors traffic anomalies (e.g., Monsoon storms) causing drive times to heavily skew, automatically correcting ETA predictions.
2. **💳 Financial Fraud Prevention:** Tracks geographical and monetary distribution shifts (e.g., overseas credit card theft) to freeze compromised models.
3. **🏭 IoT Manufacturing Quality:** Observes CNC machine sensor data (vibration and temperature) to trigger emergency halts before critical hardware failure.

---

## 🚀 The K-S Drift Detection Logic

Spectre abandons simple threshold monitoring in favor of robust non-parametric statistics. It compares the empirical cumulative distribution functions (eCDFs) of the baseline and production data. If the calculated *p-value* falls below the $0.05$ threshold, it mathematically proves the incoming data is from a foreign distribution, preemptively stopping the model from making confident but incorrect predictions.

---

## 🔧 Installation & Simulation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mutahirgillani1717/SpectreMLOps.git](https://github.com/mutahirgillani1717/SpectreMLOps.git)
   cd SpectreMLOps
2. Install dependencies:
    pip install numpy scipy scikit-learn joblib
3. Run the Live Production Simulation:
    python main.py

Author: Syed Mutahir Hussain

Academic Context: Final Year Computer Science | UET Taxila

Domain: Machine Learning Operations (MLOps), Statistical Observability, & CI/CD Infrastructure