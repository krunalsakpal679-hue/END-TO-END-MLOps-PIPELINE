import os
import numpy as np
import pytest
from src.drift_detector import DriftDetector

def test_psi_no_drift():
    detector = DriftDetector()
    np.random.seed(42)
    X_ref = np.random.normal(0, 1, size=(1000, 10))
    detector.fit_reference(X_ref)
    
    report = detector.detect_drift(X_ref)
    
    assert report.severity == 'none'
    assert report.overall_drift_detected is False
    assert len(report.triggered_features) == 0
    for i in range(10):
        assert report.per_feature[f"feature_{i}"]["psi"] < 0.05

def test_psi_severe_drift():
    detector = DriftDetector()
    np.random.seed(42)
    X_ref = np.random.normal(0, 1, size=(1000, 10))
    detector.fit_reference(X_ref)
    
    X_curr = X_ref.copy()
    X_curr[:, 0] += 3.0  # 3-sigma shift
    X_curr[:, 1] += 3.0
    X_curr[:, 2] += 3.0
    X_curr[:, 3] += 3.0
    X_curr[:, 4] += 3.0
    
    report = detector.detect_drift(X_curr)
    
    assert report.overall_drift_detected is True
    assert report.severity == 'high'
    
    for i in range(5):
        feat = f"feature_{i}"
        assert feat in report.triggered_features
        assert report.per_feature[feat]["psi"] > 0.2
        assert report.per_feature[feat]["ks_p_value"] < 0.05
        assert report.per_feature[feat]["ks_drift_detected"] is True

def test_save_load_roundtrip(tmp_path):
    detector = DriftDetector()
    np.random.seed(42)
    X_ref = np.random.normal(0, 1, size=(100, 10))
    detector.fit_reference(X_ref)
    
    file_path = tmp_path / "reference.json"
    detector.save_reference(str(file_path))
    
    detector_loaded = DriftDetector()
    detector_loaded.load_reference(str(file_path))
    
    assert detector_loaded.reference_stats.keys() == detector.reference_stats.keys()
    np.testing.assert_array_almost_equal(detector_loaded.reference_data, detector.reference_data)
