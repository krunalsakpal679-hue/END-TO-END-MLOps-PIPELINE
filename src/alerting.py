import os
import json
import time
import logging
import urllib.request
import urllib.error
import threading
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AlertManager:
    def __init__(self, slack_webhook_url: Optional[str] = None, pagerduty_key: Optional[str] = None, history_file: str = "alert_history.jsonl"):
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.pagerduty_key = pagerduty_key or os.environ.get("PAGERDUTY_KEY")
        self.history_file = history_file
        
        self.recent_alerts: Dict[str, datetime] = {}
        self.lock = threading.RLock()
        
    def _is_deduplicated(self, alert_key: str) -> bool:
        """
        Alert deduplication rule: if the same alert_key was sent less than 
        30 minutes ago, skip it silently and log: 'Alert deduplicated: <key>'.
        """
        now = datetime.now(timezone.utc)
        with self.lock:
            last_sent = self.recent_alerts.get(alert_key)
            if last_sent is not None:
                if (now - last_sent) < timedelta(minutes=30):
                    logger.info(f"Alert deduplicated: {alert_key}")
                    return True
            self.recent_alerts[alert_key] = now
            return False
            
    def _post_with_retry(self, url: str, payload: dict, headers: dict) -> bool:
        """
        Retry logic: if an HTTP call to Slack or PagerDuty fails, wait 2 seconds 
        and try again. Try up to 3 times total. If all 3 attempts fail, log an 
        error but do not crash.
        """
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    if 200 <= response.status < 300:
                        return True
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to send alert to {url} after 3 attempts. Error: {e}")
        return False
        
    def _log_history(self, alert_key: str, channel: str, message: str):
        """Append one line to alert_history.jsonl with: timestamp, alert_key, channel, message."""
        timestamp = datetime.now(timezone.utc).isoformat()
        event = {
            "timestamp": timestamp,
            "alert_key": alert_key,
            "channel": channel,
            "message": message
        }
        with self.lock:
            with open(self.history_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
                
    def _send_slack(self, alert_key: str, message: str):
        if not self.slack_webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not configured, skipping Slack alert.")
            return
            
        payload = {"text": message}
        headers = {"Content-Type": "application/json"}
        success = self._post_with_retry(self.slack_webhook_url, payload, headers)
        if success:
            self._log_history(alert_key, "slack", message)
            
    def _send_pagerduty(self, alert_key: str, message: str):
        if not self.pagerduty_key:
            logger.warning("PAGERDUTY_KEY not configured, skipping PagerDuty alert.")
            return
            
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": "trigger",
            "payload": {
                "summary": message[:1024],
                "source": "mlops_platform",
                "severity": "critical"
            }
        }
        headers = {"Content-Type": "application/json"}
        url = "https://events.pagerduty.com/v2/enqueue"
        success = self._post_with_retry(url, payload, headers)
        if success:
            self._log_history(alert_key, "pagerduty", message)

    def send_drift_alert(self, feature_name: str, psi_score: float, severity: str):
        alert_key = f"drift_{feature_name}"
        if self._is_deduplicated(alert_key):
            return
            
        message = f"Drift detected on feature {feature_name}. PSI score: {psi_score:.3f}. Severity: {severity}."
        
        self._send_slack(alert_key, message)
        if severity.lower() == 'critical':
            self._send_pagerduty(alert_key, message)
            
    def send_retrain_alert(self, trigger_reason: str, champion_f1: float, challenger_f1: float, promoted: bool):
        alert_key = "retrain_completed"
        if self._is_deduplicated(alert_key):
            return
            
        message = (f"Retraining completed (Trigger: {trigger_reason}). "
                   f"Champion F1: {champion_f1:.4f}, Challenger F1: {challenger_f1:.4f}. "
                   f"Promoted: {promoted}.")
                   
        self._send_slack(alert_key, message)
        
    def send_performance_alert(self, metric: str, current_value: float, baseline: float):
        alert_key = f"performance_{metric}"
        if self._is_deduplicated(alert_key):
            return
            
        message = f"Performance drop detected on {metric}. Current: {current_value:.4f}, Baseline: {baseline:.4f}."
        
        self._send_slack(alert_key, message)
        self._send_pagerduty(alert_key, message)
        
    def send_system_alert(self, component: str, error_message: str):
        alert_key = f"system_{component}"
        if self._is_deduplicated(alert_key):
            return
            
        truncated_error = error_message[:500]
        message = f"System error in {component}: {truncated_error}"
        
        self._send_slack(alert_key, message)
        self._send_pagerduty(alert_key, message)
        
    def get_recent_alerts(self, n: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_file):
            return []
            
        events = []
        with self.lock:
            with open(self.history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            pass
        return events[-n:] if n > 0 else events
