"""
setup.py

Installs the required packages for the Anti-Gravity project via subprocess and creates the initial MLflow experiment 
on the configured tracking server.

Constraints Met:
- Installs all required packages
- Connects to MLflow via MLFLOW_TRACKING_URI
- Creates 'anti_gravity_v1' experiment with specified tags
- Handles errors gracefully and exits with code 1 if unreachable
"""
import os
import sys
import subprocess

def main():
    requirements = [
        "mlflow>=2.10",
        "scikit-learn>=1.4",
        "pandas",
        "numpy",
        "optuna",
        "fastapi",
        "uvicorn",
        "prometheus-client",
        "pydantic>=2.0"
    ]
    
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + requirements)
        print("Packages installed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        sys.exit(1)

    try:
        import mlflow
    except ImportError:
        print("Error: mlflow could not be imported after installation.")
        sys.exit(1)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI environment variable is not set. Please set it and try again.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = "anti_gravity_v1"
    
    tags = {
        "project": "anti_gravity",
        "env": "dev",
        "owner": "mlops-engineer",
        "version": "1.0"
    }

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            print(f"Success. Experiment ID: {experiment.experiment_id}")
        else:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags=tags
            )
            print(f"Success. Experiment ID: {experiment_id}")
    except Exception as e:
        print(f"Error: Failed to connect to MLflow tracking server at '{tracking_uri}' or create experiment.")
        print(f"Details: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
