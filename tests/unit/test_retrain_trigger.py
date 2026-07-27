import os
import threading
import json
import time
from unittest.mock import patch
import pytest
from src.retrain_trigger import RetrainingOrchestrator

def test_drift_trigger():
    orchestrator = RetrainingOrchestrator()
    
    # False when only 1 feature has PSI > 0.2
    report_1_feat = {"per_feature": {"f1": {"psi": 0.25}, "f2": {"psi": 0.1}}}
    assert orchestrator.check_drift_trigger(report_1_feat) is False
    assert orchestrator.check_drift_trigger(report_1_feat) is False
    
    # True only after the same report triggers it on 2 consecutive calls
    report_2_feat = {"per_feature": {"f1": {"psi": 0.25}, "f2": {"psi": 0.25}}}
    assert orchestrator.check_drift_trigger(report_2_feat) is False
    assert orchestrator.check_drift_trigger(report_2_feat) is True
    
def test_performance_trigger():
    orchestrator = RetrainingOrchestrator()
    
    # Drop of 0.07 > 0.05
    assert orchestrator.check_performance_trigger(current_f1=0.78, baseline_f1=0.85) is True
    
    # Drop of 0.03 < 0.05
    assert orchestrator.check_performance_trigger(current_f1=0.82, baseline_f1=0.85) is False

def test_trigger_retrain_threading_and_file(tmp_path):
    history_file = tmp_path / "retrain_history.jsonl"
    orchestrator = RetrainingOrchestrator(history_file=str(history_file))
    
    def mock_run(*args, **kwargs):
        time.sleep(0.5)
        
    with patch("subprocess.run", side_effect=mock_run) as mock_subprocess:
        # Slack POST skipped gracefully when SLACK_WEBHOOK_URL is not set
        if "SLACK_WEBHOOK_URL" in os.environ:
            del os.environ["SLACK_WEBHOOK_URL"]
            
        with patch("urllib.request.urlopen") as mock_slack:
            orchestrator.trigger_retrain(reason="manual")
            mock_slack.assert_not_called()
            
        # Wait for the first mock retrain to finish to release the lock
        time.sleep(0.8)

        # Slack POST attempted when SLACK_WEBHOOK_URL is set
        os.environ["SLACK_WEBHOOK_URL"] = "http://mock.url"
        with patch("urllib.request.urlopen") as mock_slack:
            def worker():
                orchestrator.trigger_retrain(reason="drift")
                
            # Called from 2 threads simultaneously
            t1 = threading.Thread(target=worker)
            t2 = threading.Thread(target=worker)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            
            # Exactly 1 retraining job started for the simultaneous calls 
            # (total calls = 1 initial manual + 1 concurrent drift = 2)
            assert mock_subprocess.call_count == 2
            
            # Slack attempted exactly once for the successful thread
            mock_slack.assert_called_once()
            
    # retrain_history.jsonl gets one new line when trigger_retrain() is called
    # Two successful triggers occurred in this test, so exactly 2 lines total
    with open(history_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 2
    
    # get_retrain_history(5) returns a list with at most 5 items
    for _ in range(10):
        # We manually append events to simulate more history
        orchestrator._log_history("time", "simulated", "completed")
        
    history = orchestrator.get_retrain_history(n=5)
    assert len(history) <= 5
    assert len(history) == 5
    
    # Cleanup env
    del os.environ["SLACK_WEBHOOK_URL"]
