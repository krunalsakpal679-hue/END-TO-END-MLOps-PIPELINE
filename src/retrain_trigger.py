import json
import logging
import os
import subprocess
import threading
import datetime
import urllib.request
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RetrainingOrchestrator:
    def __init__(self, history_file: str = "retrain_history.jsonl"):
        self.consecutive_drift_checks = 0
        self.retraining_in_progress = False
        self.lock = threading.Lock()
        self.history_file = history_file
        self.history_lock = threading.Lock()

    def check_drift_trigger(self, drift_report: Dict[str, Any]) -> bool:
        """
        Takes a DriftReport (as a dict) and checks if the DRIFT TRIGGER condition is met:
        PSI score > 0.2 for 2 or more features simultaneously, for 2 or more consecutive checks.
        """
        features_drifting = 0
        per_feature = drift_report.get("per_feature", {})
        
        for feature_name, metrics in per_feature.items():
            if metrics.get("psi", 0) > 0.2:
                features_drifting += 1
                
        if features_drifting >= 2:
            self.consecutive_drift_checks += 1
        else:
            self.consecutive_drift_checks = 0
            
        if self.consecutive_drift_checks >= 2:
            self.consecutive_drift_checks = 0  # Reset after triggering to avoid loops
            return True
            
        return False

    def check_performance_trigger(self, current_f1: float, baseline_f1: float) -> bool:
        """
        Checks if the PERFORMANCE TRIGGER condition is met:
        Rolling F1 has dropped more than 5 percentage points below baseline.
        """
        return (baseline_f1 - current_f1) > 0.05

    def trigger_retrain(self, reason: str) -> None:
        """
        Uses a threading.Lock to prevent two retraining jobs from running simultaneously.
        Launches retrain_pipeline.py as a subprocess.
        """
        with self.lock:
            if self.retraining_in_progress:
                logger.warning("Retraining is already in progress. Ignoring trigger.")
                return
            # a) Set a flag: retraining_in_progress = True
            self.retraining_in_progress = True
            
        # Execute the rest without holding the orchestration lock to prevent blocking callers for 3600 seconds.
        # This thread will block and wait for subprocess, but other threads can still call trigger_retrain() 
        # and they will immediately return because retraining_in_progress is True.
        
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # b) Send a Slack notification
        self._send_slack_notification(f"Retraining triggered. Reason: {reason}. Time: {timestamp}")
        
        # c) Log the trigger event to retrain_history.jsonl
        self._log_history(timestamp, reason, "started")
        
        # d) Launch retrain_pipeline.py as a subprocess with a 3600-second timeout
        status = "completed"
        try:
            subprocess.run(
                ["python", "retrain_pipeline.py"], 
                timeout=3600, 
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.TimeoutExpired:
            logger.error("Retraining failed: Timeout exceeded (3600s)")
            status = "failed"
        except subprocess.CalledProcessError as e:
            logger.error(f"Retraining failed with exit code {e.returncode}: {e.stderr}")
            status = "failed"
        except Exception as e:
            logger.error(f"Retraining failed with error: {e}")
            status = "failed"
            
        # e) When done: update retrain_history.jsonl with status='completed' or 'failed'
        end_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._log_history(end_timestamp, reason, status, original_timestamp=timestamp)
        
        # f) Set retraining_in_progress = False
        with self.lock:
            self.retraining_in_progress = False

    def _send_slack_notification(self, text: str) -> None:
        """POST to the URL in environment variable SLACK_WEBHOOK_URL."""
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")
        if not webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not set, skipping Slack notification.")
            return
            
        payload = {"text": text}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(webhook_url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pass
        except Exception as e:
            logger.warning(f"Failed to send Slack notification: {e}")

    def _log_history(self, timestamp: str, reason: str, status: str, original_timestamp: str = None) -> None:
        """All file writes to retrain_history.jsonl must be thread-safe."""
        with self.history_lock:
            if original_timestamp is None:
                event = {
                    "timestamp": timestamp,
                    "reason": reason,
                    "status": status
                }
                with open(self.history_file, 'a') as f:
                    f.write(json.dumps(event) + "\n")
            else:
                if not os.path.exists(self.history_file):
                    return
                with open(self.history_file, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("timestamp") == original_timestamp:
                            event["status"] = status
                            event["end_timestamp"] = timestamp
                            lines[i] = json.dumps(event) + "\n"
                            break
                    except json.JSONDecodeError:
                        pass
                with open(self.history_file, 'w') as f:
                    f.writelines(lines)

    def get_retrain_history(self, n: int = 30) -> List[Dict[str, Any]]:
        """Reads retrain_history.jsonl and returns the last n events as a list of dicts."""
        if not os.path.exists(self.history_file):
            return []
            
        events = []
        with self.history_lock:
            with open(self.history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            pass
        return events[-n:] if n > 0 else events
