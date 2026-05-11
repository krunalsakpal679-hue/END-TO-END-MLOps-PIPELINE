import pytest
import numpy as np
from src.drift_detector import DriftDetector
from src.performance_monitor import PerformanceDriftMonitor
from tests.drift_simulator import DriftSimulator

def test_no_drift_baseline():
    X_ref, _ = DriftSimulator.generate_reference()
    detector = DriftDetector()
    detector.fit_reference(X_ref)
    
    report = detector.detect_drift(X_ref)
    assert report.overall_drift_detected is False
    assert report.severity == 'none'

def test_single_feature_shift():
    X_ref, _ = DriftSimulator.generate_reference()
    detector = DriftDetector()
    detector.fit_reference(X_ref)
    
    X_curr = DriftSimulator.inject_mean_shift(X_ref, feature_idx=2, shift_amount=4.0)
    report = detector.detect_drift(X_curr)
    
    # Needs at least 2 features to trigger overall drift
    assert report.overall_drift_detected is False
    assert report.per_feature["feature_2"]["psi"] > 0.2

def test_multi_feature_drift():
    X_ref, _ = DriftSimulator.generate_reference()
    detector = DriftDetector()
    detector.fit_reference(X_ref)
    
    X_curr = X_ref.copy()
    X_curr = DriftSimulator.inject_mean_shift(X_curr, feature_idx=0, shift_amount=4.0)
    X_curr = DriftSimulator.inject_mean_shift(X_curr, feature_idx=1, shift_amount=4.0)
    X_curr = DriftSimulator.inject_mean_shift(X_curr, feature_idx=2, shift_amount=4.0)
    
    report = detector.detect_drift(X_curr)
    assert report.overall_drift_detected is True
    # severity will be 'critical' since shift_amount=4.0 yields psi > 0.4.
    assert report.severity in ['medium', 'high', 'critical']
    assert len(report.triggered_features) == 3

def test_performance_drift_detection():
    monitor = PerformanceDriftMonitor(window_size=500, baseline_f1=0.85)
    
    # 500 samples where 80% of predictions are wrong
    for i in range(500):
        truth = 1
        pred = 0 if i < 400 else 1
        monitor.update(pred, truth)
        
    report = monitor.detect_performance_drift()
    assert report["drift_detected"] is True

def test_performance_no_false_positive():
    monitor = PerformanceDriftMonitor(window_size=500, baseline_f1=0.85)
    
    # 500 samples where 90% of predictions are correct
    for i in range(500):
        truth = 1
        pred = 1 if i < 450 else 0
        monitor.update(pred, truth)
        
    report = monitor.detect_performance_drift()
    assert report["drift_detected"] is False

def test_false_positive_rate():
    X_ref, y_ref = DriftSimulator.generate_reference(n_samples=5000, seed=42)
    detector = DriftDetector()
    detector.fit_reference(X_ref)
    
    # 1000 samples from the identical reference distribution (different seed)
    X_curr, _ = DriftSimulator.generate_reference(n_samples=1000, seed=999)
    report = detector.detect_drift(X_curr)
    
    assert report.overall_drift_detected is False
    
    # Verify label flipping accuracy explicitly to ensure 15% flip fraction works exactly
    y_flipped = DriftSimulator.inject_label_flip(y_ref, 0.15)
    num_flipped = np.sum(y_ref != y_flipped)
    expected_flips = int(len(y_ref) * 0.15)
    assert num_flipped == expected_flips
