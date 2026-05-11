import json
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from dataclasses import dataclass
from typing import Dict, List, Any, Union

@dataclass
class DriftReport:
    per_feature: Dict[str, Dict[str, Union[float, bool]]]
    overall_drift_detected: bool
    severity: str
    triggered_features: List[str]

class DriftDetector:
    def __init__(self) -> None:
        self.reference_stats: Dict[str, Any] = {}
        self.reference_data: Union[np.ndarray, None] = None

    def fit_reference(self, X: np.ndarray) -> None:
        """Fits the reference statistics based on the training data."""
        self.reference_data = X.copy()
        self.reference_stats = {}
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            # Remove NaNs if any
            valid_data = feature_data[~np.isnan(feature_data)]
            hist, bin_edges = np.histogram(valid_data, bins=10)
            self.reference_stats[f"feature_{i}"] = {
                "mean": float(np.mean(valid_data)),
                "std": float(np.std(valid_data)),
                "min": float(np.min(valid_data)),
                "max": float(np.max(valid_data)),
                "hist": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }

    def compute_psi(self, feature_idx: int, current_data: np.ndarray) -> float:
        """Computes Population Stability Index for a specific feature."""
        if self.reference_data is None:
            raise ValueError("Reference data is not fitted.")
            
        ref_data = self.reference_data[:, feature_idx]
        ref_valid = ref_data[~np.isnan(ref_data)]
        curr_valid = current_data[~np.isnan(current_data)]
        
        ref_hist, edges = np.histogram(ref_valid, bins=10, density=True)
        cur_hist, _ = np.histogram(curr_valid, bins=edges, density=True)
        
        # Clip to avoid log(0) which gives infinity
        ref_hist = np.clip(ref_hist, 1e-10, None)
        cur_hist = np.clip(cur_hist, 1e-10, None)
        
        psi = np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist))
        return float(psi)

    def compute_ks_test(self, feature_idx: int, current_data: np.ndarray) -> Dict[str, Union[float, bool]]:
        """Computes Kolmogorov-Smirnov test for a specific feature."""
        if self.reference_data is None:
            raise ValueError("Reference data is not fitted.")
            
        ref_data = self.reference_data[:, feature_idx]
        ref_valid = ref_data[~np.isnan(ref_data)]
        curr_valid = current_data[~np.isnan(current_data)]
        
        stat, p_value = stats.ks_2samp(ref_valid, curr_valid)
        return {
            "statistic": float(stat),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05)
        }

    def compute_js_divergence(self, feature_idx: int, current_data: np.ndarray) -> float:
        """Computes Jensen-Shannon divergence for a specific feature."""
        ref_stat = self.reference_stats[f"feature_{feature_idx}"]
        bin_edges = np.array(ref_stat["bin_edges"])
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        ref_hist = np.array(ref_stat["hist"])
        
        valid_data = current_data[~np.isnan(current_data)]
        curr_hist, _ = np.histogram(valid_data, bins=bin_edges)
        
        ref_frac = ref_hist / np.sum(ref_hist)
        if np.sum(curr_hist) == 0:
            curr_frac = np.ones_like(curr_hist, dtype=float) / len(curr_hist)
        else:
            curr_frac = curr_hist / np.sum(curr_hist)
        
        epsilon = 1e-4
        ref_frac = np.where(ref_frac == 0, epsilon, ref_frac)
        curr_frac = np.where(curr_frac == 0, epsilon, curr_frac)
        
        ref_frac = ref_frac / np.sum(ref_frac)
        curr_frac = curr_frac / np.sum(curr_frac)
        
        js_dist = jensenshannon(ref_frac, curr_frac, base=2)
        return float(js_dist)

    def detect_drift(self, X_current: np.ndarray) -> DriftReport:
        """Runs all drift tests for all features and returns a report."""
        per_feature = {}
        triggered_features = []
        
        for i in range(X_current.shape[1]):
            feat_name = f"feature_{i}"
            current_data = X_current[:, i]
            
            psi = self.compute_psi(i, current_data)
            ks = self.compute_ks_test(i, current_data)
            js = self.compute_js_divergence(i, current_data)
            
            per_feature[feat_name] = {
                "psi": psi,
                "ks_statistic": ks["statistic"],
                "ks_p_value": ks["p_value"],
                "ks_drift_detected": ks["drift_detected"],
                "js_divergence": js
            }
            
            if psi > 0.2:
                triggered_features.append(feat_name)
                
        num_triggered = len(triggered_features)
        overall_drift_detected = num_triggered >= 2
        
        any_critical = any(per_feature[f]["psi"] > 0.4 for f in per_feature)
        
        if num_triggered >= 4:
            severity = 'high'
        elif any_critical:
            severity = 'critical'
        elif num_triggered >= 2:
            severity = 'medium'
        elif num_triggered == 1:
            severity = 'low'
        else:
            severity = 'none'
            
        return DriftReport(
            per_feature=per_feature,
            overall_drift_detected=overall_drift_detected,
            severity=severity,
            triggered_features=triggered_features
        )

    def save_reference(self, file_path: str) -> None:
        """Saves reference statistics to a JSON file."""
        data = {
            "reference_stats": self.reference_stats,
            "reference_data": self.reference_data.tolist() if self.reference_data is not None else None
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def load_reference(self, file_path: str) -> None:
        """Loads reference statistics from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.reference_stats = data["reference_stats"]
        if data.get("reference_data") is not None:
            self.reference_data = np.array(data["reference_data"])
        else:
            self.reference_data = None
