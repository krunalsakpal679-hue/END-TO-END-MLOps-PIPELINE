"""
registry_manager.py

Module to manage the MLflow Model Registry lifecycle for regulated ML systems.
This module enforces model governance by strictly requiring metric thresholds before
allowing promotion to the Production stage.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

# Configure logging to use stdout and a standard timestamp format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RegistryManager:
    """
    Manages the lifecycle of models within the MLflow Model Registry.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initializes the RegistryManager with an MlflowClient.

        Args:
            tracking_uri (Optional[str]): The URI of the MLflow tracking server.
                                          Defaults to MLFLOW_TRACKING_URI environment variable.
        """
        if tracking_uri is None:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def register_model(self, run_id: str, model_name: str, description: str) -> ModelVersion:
        """
        Registers a model from a specific run to the MLflow Model Registry.

        Args:
            run_id (str): The MLflow run ID containing the model artifact.
            model_name (str): The name to register the model under.
            description (str): Description to attach to the registered model version.

        Returns:
            ModelVersion: The registered model version object.
        """
        # Ensure the registered model container exists first
        try:
            self.client.create_registered_model(model_name)
        except Exception:
            # Container already exists, we can safely proceed
            pass

        # MLflow direct client registration expects a source path.
        # mlflow.sklearn.log_model uses the artifact path "model" by default.
        source = f"runs:/{run_id}/model"
        
        model_version = self.client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id
        )
        
        # Add description to the newly created model version
        self.client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        logger.info("Registered model %s as version %s", model_name, model_version.version)
        return model_version

    def promote_to_staging(self, model_name: str, version: str, notes: str) -> None:
        """
        Promotes a specific model version to the 'Staging' stage.

        Args:
            model_name (str): The registered model name.
            version (str): The model version number as a string.
            notes (str): A comment to add regarding this transition.
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging",
            archive_existing_versions=False
        )
        
        # Open source MLflow does not natively have transition-specific comments.
        # Best practice is to append the notes as a tag on the model version itself.
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="staging_notes",
            value=notes
        )
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("[%s] Promoted %s v%s to Staging stage.", current_time, model_name, version)

    def promote_to_production(self, model_name: str, version: str, min_f1: float = 0.85, min_auc: float = 0.90) -> None:
        """
        Promotes a model version to 'Production' if it meets performance thresholds.

        Args:
            model_name (str): The registered model name.
            version (str): The model version number as a string.
            min_f1 (float): The minimum required f1_weighted score.
            min_auc (float): The minimum required roc_auc score.

        Raises:
            ValueError: If the model's metrics do not meet the minimum thresholds.
        """
        # Fetch the model version to get its run_id
        model_version = self.client.get_model_version(model_name, version)
        run_id = model_version.run_id
        
        # Fetch run details to get the logged metrics
        run = self.client.get_run(run_id)
        metrics = run.data.metrics
        
        f1 = metrics.get("f1_weighted", 0.0)
        auc = metrics.get("roc_auc", 0.0)
        
        # Enforcement checks
        if f1 < min_f1:
            raise ValueError(f"Model fails threshold: f1_weighted ({f1}) < min_f1 ({min_f1})")
            
        if auc < min_auc:
            raise ValueError(f"Model fails threshold: roc_auc ({auc}) < min_auc ({min_auc})")
            
        # All checks passed, transition to Production and archive previous Production models
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        logger.info("Promoted %s v%s to Production. Previous version archived.", model_name, version)

    def get_production_uri(self, model_name: str) -> str:
        """
        Returns the URI string for loading the current Production model.

        Args:
            model_name (str): The registered model name.

        Returns:
            str: The URI string pointing to the Production model.
        """
        return f"models:/{model_name}/Production"

    def generate_registry_report(self, model_name: str) -> None:
        """
        Generates and logs a Markdown-formatted table of all versions for a given model.

        Args:
            model_name (str): The registered model name.
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
        except Exception as e:
            logger.error("Failed to search model versions: %s", e)
            return

        if not versions:
            logger.info("No versions found for model '%s'.", model_name)
            return

        # Initialize markdown table
        report_lines = [
            f"\nRegistry Report for Model: {model_name}",
            "| Version | Stage | F1 Weighted | ROC AUC | Creation Date |",
            "|---------|-------|-------------|---------|---------------|"
        ]

        # Sort chronologically or by version number safely
        versions = sorted(versions, key=lambda v: int(v.version))

        for v in versions:
            run_id = v.run_id
            
            # Fetch the actual metrics from the underlying run
            try:
                run = self.client.get_run(run_id)
                f1 = run.data.metrics.get("f1_weighted", "N/A")
                if isinstance(f1, float):
                    f1 = f"{f1:.4f}"
                
                auc = run.data.metrics.get("roc_auc", "N/A")
                if isinstance(auc, float):
                    auc = f"{auc:.4f}"
            except Exception:
                f1, auc = "Error", "Error"

            # Parse creation_timestamp (milliseconds to formatted string)
            if v.creation_timestamp:
                dt = datetime.fromtimestamp(v.creation_timestamp / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
            else:
                dt = "Unknown"

            report_lines.append(f"| {v.version} | {v.current_stage} | {f1} | {auc} | {dt} |")

        report = "\n".join(report_lines)
        logger.info(report)

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "report":
        model_name_arg = sys.argv[2]
        manager = RegistryManager()
        manager.generate_registry_report(model_name_arg)
    else:
        logger.warning("Usage: python registry_manager.py report <model_name>")
