import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model(f1_mean, f1_std, f2_mean, f2_std):
    """Dynamically generates baseline data based on UI parameters."""
    print(f"[Spectre-Core] 🔄 Training Baseline AI... Features -> F1({f1_mean}), F2({f2_mean})")
    
    # Generate dynamic feature columns
    f1_data = np.random.normal(loc=f1_mean, scale=f1_std, size=(1000, 1))
    f2_data = np.random.normal(loc=f2_mean, scale=f2_std, size=(1000, 1))
    
    X_train = np.hstack((f1_data, f2_data))
    
    # Dummy target logic (e.g., 1 for acceptable range, 0 for outlier)
    y_train = (f2_data < (f2_mean + f2_std)).astype(int) 

    # Train the AI
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # Save the compiled model and Baseline Data
    joblib.dump(model, "spectre_model.pkl")
    np.save("baseline_X.npy", X_train)

    print("[Spectre-Core] ✅ AI Model and Baseline Data saved to disk.\n")
    return True