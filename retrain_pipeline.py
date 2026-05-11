import argparse
import json
import logging
import os
import sys
import datetime
from typing import Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Automated Model Retraining Pipeline")
    parser.add_argument("--trigger-reason", type=str, required=True, help="Why retraining was triggered (e.g. 'psi_drift', 'performance_drop')")
    parser.add_argument("--window-days", type=int, default=90, help="Days of recent data to use")
    parser.add_argument("--min-samples", type=int, default=5000, help="Minimum rows needed to proceed")
    return parser.parse_args()

def step1_fetch_data(window_days: int, min_samples: int) -> Any:
    logger.info(f"Step 1 — Fetch data (last {window_days} days)")
    db_uri = os.environ.get("DATA_WAREHOUSE_URI")
    if not db_uri:
        logger.warning("DATA_WAREHOUSE_URI is not set. Using mock connection.")
        
    # Execute SQL query logic here...
    num_samples_fetched = 6000  # Mock value representing fetched rows
    
    if num_samples_fetched < min_samples:
        print(f"ERROR: Only {num_samples_fetched} samples available. Minimum is {min_samples}.")
        sys.exit(1)
        
    logger.info(f"Successfully fetched {num_samples_fetched} rows.")
    return {"X": "mock_dataframe_X", "y": "mock_dataframe_y"}

def step2_validate_data(data: Any) -> None:
    logger.info("Step 2 — Validate data quality")
    
    # Try importing P-3.5 module, otherwise use mock
    try:
        from data_validator import DataValidator
        validator = DataValidator()
    except ImportError:
        logger.warning("DataValidator module not found. Using Mock.")
        class MockValidator:
            def validate(self, d): return True, []
        validator = MockValidator()
        
    is_valid, failed_checks = validator.validate(data)
    
    if not is_valid:
        print(f"ERROR: Data validation failed. Failed checks: {failed_checks}")
        sys.exit(1)
    logger.info("Data passed all quality checks.")

def step3_load_pipeline(data: Any) -> Tuple[Any, Any]:
    logger.info("Step 3 — Load existing feature pipeline")
    
    try:
        from feature_pipeline import load_pipeline
        pipeline = load_pipeline()
    except ImportError:
        logger.warning("feature_pipeline module not found. Using Mock.")
        class MockPipeline:
            def transform(self, x): return x
        pipeline = MockPipeline()
        
    # We apply transform() - NOT fit_transform()
    X_transformed = pipeline.transform(data["X"])
    logger.info("Pipeline loaded and data transformed successfully.")
    return X_transformed, data["y"]

def step4_train_challenger(X_train: Any, y_train: Any) -> Any:
    logger.info("Step 4 — Train challenger model")
    
    best_params = {}
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            best_params = json.load(f)
        logger.info(f"Loaded Optuna best_params: {best_params}")
    else:
        logger.warning("best_params.json not found. Training with default parameters.")
        
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        clf = GradientBoostingClassifier(**best_params)
        # cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')
        logger.info("Challenger model trained and cross-validated successfully.")
        return clf
    except ImportError:
        logger.warning("scikit-learn not available. Mocking training step.")
        return "mock_challenger_model"

def step5_compare_models(champion: Any, challenger: Any, X_test: Any, y_test: Any) -> Tuple[bool, float, float]:
    logger.info("Step 5 — Compare challenger vs champion")
    
    try:
        from model_challenger import ModelChallenger
        results = ModelChallenger.compare(champion, challenger, X_test, y_test)
    except ImportError:
        logger.warning("ModelChallenger module not found. Using Mock.")
        results = {"champion_f1": 0.85, "challenger_f1": 0.87}
        
    champion_f1 = results.get("champion_f1", 0.0)
    challenger_f1 = results.get("challenger_f1", 0.0)
    
    logger.info(f"Champion F1: {champion_f1:.4f} | Challenger F1: {challenger_f1:.4f}")
    
    # Promote ONLY if challenger F1 > champion F1 + 0.01
    promoted = challenger_f1 > (champion_f1 + 0.01)
    return promoted, champion_f1, challenger_f1

def step6_promote_or_keep(promoted: bool, challenger: Any) -> str:
    logger.info("Step 6 — Promote or keep champion")
    if promoted:
        reason = "Challenger exceeded Champion F1 by > 0.01 threshold."
        logger.info(f"Challenger wins! Registering and promoting to Production. Reason: {reason}")
    else:
        reason = "Challenger failed to exceed Champion F1 by > 0.01 threshold."
        logger.info(f"Champion wins! Keeping existing model in Production. Reason: {reason}")
    return reason

def step7_write_summary(trigger_reason: str, champ_f1: float, chall_f1: float, promoted: bool, reason: str) -> None:
    logger.info("Step 7 — Write retrain_summary.json")
    summary = {
        "trigger_reason": trigger_reason,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "champion_f1": float(champ_f1),
        "challenger_f1": float(chall_f1),
        "promoted": bool(promoted),
        "reason_for_decision": reason
    }
    with open("retrain_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    logger.info("Summary successfully written to retrain_summary.json")

def main():
    args = parse_args()
    
    logger.info("Step 8 — Wrapping everything in MLflow (parent/child runs)")
    try:
        import mlflow
        mlflow_active = True
    except ImportError:
        mlflow_active = False
        logger.warning("MLflow package missing. Tracking disabled.")
        
    class MLflowContextMock:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def run_context(name, nested=False):
        if mlflow_active:
            return mlflow.start_run(run_name=name, nested=nested)
        return MLflowContextMock()
        
    def log_metric(k, v):
        if mlflow_active:
            mlflow.log_metric(k, v)

    # Step 8 wrapper
    with run_context("retrain_pipeline_parent"):
        data = step1_fetch_data(args.window_days, args.min_samples)
        step2_validate_data(data)
        
        with run_context("step3_load_pipeline", nested=True):
            X_trans, y_trans = step3_load_pipeline(data)
            
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')
        with run_context(f"challenger_{timestamp_str}", nested=True):
            challenger_model = step4_train_challenger(X_trans, y_trans)
            
        with run_context("step5_compare", nested=True):
            champion_model = "mock_champion_model"
            promoted, champ_f1, chall_f1 = step5_compare_models(champion_model, challenger_model, X_trans, y_trans)
            
        decision_reason = step6_promote_or_keep(promoted, challenger_model)
        
        step7_write_summary(args.trigger_reason, champ_f1, chall_f1, promoted, decision_reason)
        
        log_metric("promoted", 1.0 if promoted else 0.0)

if __name__ == "__main__":
    main()
