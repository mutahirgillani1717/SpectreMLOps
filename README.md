# 📡 Spectre MLOps: Automated Model Observability Pipeline

**Spectre** is a production-grade Machine Learning Operations (MLOps) pipeline designed to monitor live AI models, detect statistical data drift in real-time, and trigger automated self-healing (retraining) loops to maintain predictive accuracy in volatile environments.

---

## 🧠 Core Architecture

* **The Predictor (`model_pipeline.py`):** Handles the initial training and deployment of the base model (Random Forest) while securing the "Baseline Distribution" as a mathematical point of reference.
* **The Observer (`drift_detector.py`):** Acts as the system's immune system. It continuously analyzes incoming production data streams using the **Two-Sample Kolmogorov-Smirnov (K-S) Statistical Test**.
* **The Orchestrator (`main.py`):** Simulates a live production server. It routes traffic to the observer, catches `[CRITICAL ALERT]` drift notifications (where $p < 0.05$), and dynamically triggers the CI/CD retraining loop to redeploy a fresh model.

---

## 🚀 The K-S Drift Detection Logic

Unlike simple threshold monitors, Spectre uses non-parametric statistics to compare the empirical cumulative distribution functions (eCDFs) of the baseline and production data. If the calculated *p-value* falls below the $0.05$ threshold, it mathematically proves the incoming data is from a foreign distribution, preemptively stopping the model from making confident but incorrect predictions.

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