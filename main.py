import numpy as np
import time
from drift_detector import DriftDetector
from model_pipeline import train_and_save_model

def simulate_production_stream():
    print("--- 🚀 SPECTRE MLOPS: Automated Observability Pipeline ---")
    
    # 1. Initial Deployment (Day 1)
    train_and_save_model()
    detector = DriftDetector()
    
    # 2. Simulate Normal Traffic (Data similar to training)
    print("📡 [Live Server] Simulating Week 1: Normal incoming traffic...")
    normal_traffic = np.random.normal(loc=0.0, scale=1.0, size=(200, 2))
    time.sleep(1.5)
    
    is_drifting, report = detector.check_drift(normal_traffic)
    if not is_drifting:
        print(f"📊 [Observability] Status: HEALTHY. No data drift detected. (P-Value: {report['Feature_0']['p_value']:.3f})")
        
    # 3. Simulate Data Drift (e.g., market crash, broken hardware sensor)
    print("\n📡 [Live Server] Simulating Week 2: Sudden data distribution shift (Anomaly injected)...")
    time.sleep(2)
    
    # The mean shifts completely off the charts (from 0.0 to 2.5)
    drifted_traffic = np.random.normal(loc=2.5, scale=1.5, size=(200, 2)) 
    
    is_drifting, report = detector.check_drift(drifted_traffic)
    
    if is_drifting:
        print(f"⚠️ [CRITICAL ALERT] Data Drift Detected! Statistical distribution mismatch.")
        print(f"   -> Feature 0 P-Value: {report['Feature_0']['p_value']:.5e} (Below 0.05 threshold)")
        print(f"   -> Feature 1 P-Value: {report['Feature_1']['p_value']:.5e} (Below 0.05 threshold)")
        
        print("\n⚙️ [Orchestrator] Model accuracy is compromised. Triggering automated retraining loop...")
        time.sleep(2)
        
        # 4. Automated Self-Healing Trigger
        train_and_save_model()
        print("✅ [Orchestrator] New optimized model deployed to production. System stabilized.")

if __name__ == "__main__":
    simulate_production_stream()