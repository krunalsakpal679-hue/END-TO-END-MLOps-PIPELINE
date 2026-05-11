import os
import threading
import pytest
from src.performance_monitor import PerformanceDriftMonitor

def test_thread_safety():
    monitor = PerformanceDriftMonitor(window_size=1000)
    
    def worker():
        for _ in range(50):
            monitor.update(prediction=1, ground_truth=1)
            
    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    # 20 threads * 50 updates = 1000 items precisely. 
    # Validates updates are perfectly thread safe without dropped instances.
    assert len(monitor.window) == 1000

def test_drift_detection():
    monitor = PerformanceDriftMonitor(window_size=1000, baseline_f1=0.85)
    
    # Verify RuntimeError when window < 100 samples
    for _ in range(99):
        monitor.update(1, 1)
    with pytest.raises(RuntimeError):
        monitor.compute_rolling_metrics()
        
    monitor.window.clear()
    
    # Validate exactly 80% wrong predictions (80 wrong / 20 correct)
    for _ in range(80):
        monitor.update(prediction=1, ground_truth=0)  # wrong
    for _ in range(20):
        monitor.update(prediction=1, ground_truth=1)  # correct
        
    report = monitor.detect_performance_drift()
    assert report["drift_detected"] is True

def test_no_drift():
    monitor = PerformanceDriftMonitor(window_size=1000, baseline_f1=0.85)
    
    # Validate exactly 90% correct predictions (90 correct / 10 wrong)
    for _ in range(90):
        monitor.update(prediction=1, ground_truth=1)  # correct
    for _ in range(10):
        monitor.update(prediction=1, ground_truth=0)  # wrong
        
    report = monitor.detect_performance_drift()
    assert report["drift_detected"] is False
    
    # Ensure all 4 key metric names are returned perfectly
    metrics = monitor.compute_rolling_metrics()
    assert set(metrics.keys()) == {"accuracy", "f1_weighted", "precision", "recall"}

def test_save_load_roundtrip(tmp_path):
    monitor = PerformanceDriftMonitor(window_size=1000, baseline_f1=0.85)
    
    # Validate deque respects maxlen precisely (add 1001 items to 1000 maxlen deque)
    for i in range(1001):
        monitor.update(prediction=1, ground_truth=i % 2, timestamp=f"ts_{i}")
        
    assert len(monitor.window) == 1000
    assert monitor.window[0][2] == "ts_1"  # Oldest item "ts_0" is successfully dropped
    
    # Test JSON save and load capabilities
    file_path = tmp_path / "perf_state.json"
    monitor.save_state(str(file_path))
    
    loaded_monitor = PerformanceDriftMonitor()
    loaded_monitor.load_state(str(file_path))
    
    assert len(loaded_monitor.window) == 1000
    assert loaded_monitor.baseline_f1 == 0.85
    assert loaded_monitor.window.maxlen == 1000
    assert list(loaded_monitor.window) == list(monitor.window)
