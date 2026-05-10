"""
optuna_search.py

Bayesian hyperparameter search for GradientBoostingClassifier using Optuna.
Runs 50 trials (configurable) while evaluating via 5-fold cross-validation.
Integrates tightly with MLflow, logging all trials as nested runs under 'optuna_search'.
"""

import os
import sys
import json
import argparse
import subprocess

# Auto-install dependencies if missing
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("Installing optuna, plotly, scikit-learn (if needed)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna", "plotly", "scikit-learn"])
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import mlflow

def load_data():
    data_path = os.path.join("data", "train.csv")
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run the data generation script or provide the dataset.")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    expected_cols = [f"feat_{i:02d}" for i in range(1, 11)]
    X = df[expected_cols]
    y = df['label']
    
    # 70% train
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    return X_train, y_train

def main():
    parser = argparse.ArgumentParser(description="Optuna Bayesian Search for GradientBoosting")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    
    X_train, y_train = load_data()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "anti_gravity_v1"
    mlflow.set_experiment(experiment_name)
    
    def objective(trial):
        # The search space mapping exactly to specifications
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.5, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'random_state': 42  # Explicitly constrained to 42
        }
        
        # Log trial as a nested MLflow run
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            try:
                model = GradientBoostingClassifier(**params)
                
                # Evaluate with 5-fold cross-validation on the training set using n_jobs=-1
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
                mean_auc = cv_scores.mean()
                
                # Log hyperparameters and metrics
                loggable_params = {k: v for k, v in params.items() if k != 'random_state'}
                mlflow.log_params(loggable_params)
                mlflow.log_metric("roc_auc", mean_auc)
                
                # Report to optuna for MedianPruner checks
                # We simulate multiple steps to trigger the pruner (since we just have one cv_score.mean())
                for step in range(6):
                    trial.report(mean_auc, step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
                return mean_auc
                
            except optuna.exceptions.TrialPruned:
                # If pruned, add the tag
                mlflow.set_tag("pruned", "true")
                raise

    # TPESampler with seed=42 and MedianPruner configured as requested
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    
    # Wrap all trials in a single parent MLflow run named 'optuna_search'
    with mlflow.start_run(run_name="optuna_search"):
        # Run sequentially (n_jobs=1), with 3600 timeout
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, timeout=3600)
        
        # Save best params to JSON
        best_params = study.best_params
        with open("best_params.json", "w") as f:
            json.dump(best_params, f, indent=4)
            
        # Create and log optimization history plot via Matplotlib to avoid Kaleido dependencies
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import plot_optimization_history
        
        ax = plot_optimization_history(study)
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("optimization_history.png")
        plt.close(fig)
        
        mlflow.log_artifact("optimization_history.png")
        
    print(f"Best AUC: {study.best_value:.4f}. Best params saved to best_params.json")
    
if __name__ == "__main__":
    main()
