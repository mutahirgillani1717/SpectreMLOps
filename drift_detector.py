import numpy as np
from scipy.stats import ks_2samp

class DriftDetector:
    def __init__(self, baseline_path="baseline_X.npy", p_value_threshold=0.05):
        """Initializes the monitor with the baseline training distribution."""
        self.baseline_data = np.load(baseline_path)
        self.p_value_threshold = p_value_threshold

    def check_drift(self, incoming_data):
        """
        Uses the Kolmogorov-Smirnov statistic to check if the incoming data
        distribution has drifted significantly from the baseline.
        """
        num_features = self.baseline_data.shape[1]
        drift_detected = False
        report = {}

        for i in range(num_features):
            baseline_feature = self.baseline_data[:, i]
            incoming_feature = incoming_data[:, i]
            
            # Perform the KS Test
            stat, p_value = ks_2samp(baseline_feature, incoming_feature)
            
            # If p_value is less than 0.05, we reject the null hypothesis (Drift Occurred!)
            is_drifting = p_value < self.p_value_threshold
            report[f"Feature_{i}"] = {"p_value": p_value, "drifting": is_drifting}
            
            if is_drifting:
                drift_detected = True

        return drift_detected, report