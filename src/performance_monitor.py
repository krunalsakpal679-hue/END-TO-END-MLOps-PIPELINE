import json
import threading
from collections import deque
from typing import Dict, Any, Optional
import numpy as np

class PerformanceDriftMonitor:
    def __init__(self, window_size: int = 1000, baseline_f1: float = 0.85) -> None:
        self.baseline_f1 = baseline_f1
        self.window = deque(maxlen=window_size)
        self.lock = threading.RLock()

    def update(self, prediction: int, ground_truth: int, timestamp: Optional[str] = None) -> None:
        """Adds one (prediction, ground_truth, timestamp) tuple to the rolling window."""
        with self.lock:
            self.window.append((prediction, ground_truth, timestamp))

    def compute_rolling_metrics(self) -> Dict[str, float]:
        """Computes accuracy, f1_weighted, precision, and recall on the window."""
        with self.lock:
            if len(self.window) < 100:
                raise RuntimeError("Not enough samples to compute reliable metrics. Need at least 100.")
            
            y_pred = np.array([item[0] for item in self.window])
            y_true = np.array([item[1] for item in self.window])
            
            accuracy = np.mean(y_pred == y_true)
            
            labels = np.unique(np.concatenate([y_true, y_pred]))
            
            f1_scores = []
            supports = []
            precisions = []
            recalls = []
            
            for label in labels:
                tp = np.sum((y_pred == label) & (y_true == label))
                fp = np.sum((y_pred == label) & (y_true != label))
                fn = np.sum((y_pred != label) & (y_true == label))
                
                support = np.sum(y_true == label)
                supports.append(support)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                
            total_support = np.sum(supports)
            if total_support > 0:
                f1_weighted = sum(f1 * sup for f1, sup in zip(f1_scores, supports)) / total_support
                precision_weighted = sum(p * sup for p, sup in zip(precisions, supports)) / total_support
                recall_weighted = sum(r * sup for r, sup in zip(recalls, supports)) / total_support
            else:
                f1_weighted = 0.0
                precision_weighted = 0.0
                recall_weighted = 0.0
                
            return {
                "accuracy": float(accuracy),
                "f1_weighted": float(f1_weighted),
                "precision": float(precision_weighted),
                "recall": float(recall_weighted)
            }

    def detect_performance_drift(self) -> Dict[str, Any]:
        """Detects if F1 has dropped significantly compared to baseline."""
        with self.lock:
            metrics = self.compute_rolling_metrics()
            current_f1 = metrics["f1_weighted"]
            baseline = self.baseline_f1
            f1_delta = current_f1 - baseline
            drift_detected = f1_delta < -0.05
            
            msg = (f"Performance drift detected. F1 dropped by {-f1_delta:.4f} "
                   f"(baseline {baseline}, current {current_f1:.4f}).") if drift_detected else "Model performance is stable."
            
            return {
                "current_f1": float(current_f1),
                "baseline_f1": float(baseline),
                "f1_delta": float(f1_delta),
                "drift_detected": bool(drift_detected),
                "alert_message": msg
            }

    def save_state(self, file_path: str) -> None:
        """Saves the current window data and baseline to a JSON file."""
        with self.lock:
            data = {
                "window_size": self.window.maxlen,
                "baseline_f1": self.baseline_f1,
                "window": list(self.window)
            }
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def load_state(self, file_path: str) -> None:
        """Restores the monitor state from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        with self.lock:
            self.baseline_f1 = data.get("baseline_f1", self.baseline_f1)
            maxlen = data.get("window_size", 1000)
            
            window_data = [tuple(item) for item in data.get("window", [])]
            self.window = deque(window_data, maxlen=maxlen)
