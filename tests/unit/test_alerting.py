import os
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from src.alerting import AlertManager

def test_slack_post_format(tmp_path):
    history_file = tmp_path / "alerts.jsonl"
    manager = AlertManager(slack_webhook_url="http://mock.slack", history_file=str(history_file))
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        manager.send_drift_alert("feat_1", 0.25, "medium")
        
        # 1. verify HTTP POST was made
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://mock.slack"
        
        # 2. verify JSON body formatting
        body = json.loads(req.data.decode("utf-8"))
        assert "text" in body
        assert "Drift detected on feature feat_1. PSI score: 0.250. Severity: medium." in body["text"]

def test_deduplication_and_cooldown(tmp_path):
    history_file = tmp_path / "alerts.jsonl"
    manager = AlertManager(slack_webhook_url="http://mock.slack", history_file=str(history_file))
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        # Call 1 - Time T
        # Since we just need to shift time, we can mock datetime
        import src.alerting
        
        with patch("src.alerting.datetime") as mock_datetime:
            t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            mock_datetime.now.return_value = t0
            manager.send_drift_alert("feat_1", 0.25, "medium")
            
            # Call 2 - Time T + 10 mins (within 30m window)
            t1 = t0 + timedelta(minutes=10)
            mock_datetime.now.return_value = t1
            manager.send_drift_alert("feat_1", 0.25, "medium")
            
            # Verify only 1 POST was made
            assert mock_urlopen.call_count == 1
            
            # Call 3 - Time T + 35 mins (cooldown expired)
            t2 = t0 + timedelta(minutes=35)
            mock_datetime.now.return_value = t2
            manager.send_drift_alert("feat_1", 0.25, "medium")
            
            # Verify 2nd POST was made
            assert mock_urlopen.call_count == 2

def test_system_alert_truncation(tmp_path):
    history_file = tmp_path / "alerts.jsonl"
    manager = AlertManager(slack_webhook_url="http://mock.slack", history_file=str(history_file))
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        long_error = "A" * 1000
        manager.send_system_alert("database", long_error)
        
        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data.decode("utf-8"))
        # System error in database: + 500 chars limit
        assert len(body["text"]) < 550
        assert "A" * 500 in body["text"]
        assert "A" * 501 not in body["text"]

def test_slack_retry_logic(tmp_path):
    history_file = tmp_path / "alerts.jsonl"
    manager = AlertManager(slack_webhook_url="http://mock.slack", history_file=str(history_file))
    
    with patch("urllib.request.urlopen") as mock_urlopen, patch("time.sleep") as mock_sleep:
        # Simulate 500 error / HTTP Error throwing exception 3 times
        mock_urlopen.side_effect = urllib.error.HTTPError("url", 500, "Internal Server Error", {}, None)
        
        manager.send_drift_alert("feat_1", 0.25, "medium")
        
        # Called exactly 3 times before giving up
        assert mock_urlopen.call_count == 3
        # Sleep called twice (after attempt 1 and 2)
        assert mock_sleep.call_count == 2
        # No crash occurred!

def test_alert_history_line_and_fetch(tmp_path):
    history_file = tmp_path / "alerts.jsonl"
    manager = AlertManager(slack_webhook_url="http://mock.slack", history_file=str(history_file))
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response
        
        import src.alerting
        with patch("src.alerting.datetime") as mock_datetime:
            # Generate 10 valid alerts
            for i in range(10):
                t = datetime(2025, 1, 1, 12 + i, 0, 0, tzinfo=timezone.utc)
                mock_datetime.now.return_value = t
                manager.send_drift_alert(f"feat_{i}", 0.25, "medium")
                
    # 1 line per successfully sent alert
    with open(history_file, 'r') as f:
        lines = f.readlines()
    assert len(lines) == 10
    
    # get_recent_alerts(5) returns at most 5
    recent = manager.get_recent_alerts(5)
    assert len(recent) == 5

def test_warning_on_missing_url(tmp_path, caplog):
    history_file = tmp_path / "alerts.jsonl"
    # Do NOT configure slack
    manager = AlertManager(history_file=str(history_file))
    
    with caplog.at_level(logging.WARNING):
        manager.send_drift_alert("feat_1", 0.25, "medium")
        
    assert "SLACK_WEBHOOK_URL not configured" in caplog.text
    # Ensure it returns without crashing
    assert True
