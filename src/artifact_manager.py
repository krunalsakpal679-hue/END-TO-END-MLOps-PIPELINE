"""
artifact_manager.py

Utilities to inspect, clean, and manage MLflow runs and artifacts safely.
Implements hard safety constraints (dry-run defaults) for destructive operations.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

class ArtifactManager:
    def __init__(self, tracking_uri: str = None):
        if tracking_uri is None:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def _get_run_artifact_size_bytes(self, run_id: str, path: str = "") -> int:
        """Recursively calculate the total size of a run's artifacts in bytes."""
        total_size = 0
        try:
            artifacts = self.client.list_artifacts(run_id, path)
            for file_info in artifacts:
                if file_info.is_dir:
                    total_size += self._get_run_artifact_size_bytes(run_id, file_info.path)
                else:
                    total_size += file_info.file_size if file_info.file_size else 0
        except Exception:
            # Handle potential connection issues or missing artifacts gracefully
            pass
        return total_size

    def list_large_artifacts(self, experiment_id: str, min_mb: float = 100.0) -> List[Dict[str, Any]]:
        """Scans all runs in an experiment and identifies runs with large artifacts."""
        runs = self.client.search_runs([experiment_id])
        large_runs = []
        
        for run in runs:
            size_bytes = self._get_run_artifact_size_bytes(run.info.run_id)
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb >= min_mb:
                large_runs.append({
                    'run_id': run.info.run_id,
                    'run_name': run.data.tags.get("mlflow.runName", "Unnamed"),
                    'total_artifact_size_mb': size_mb
                })
                
        print(f"\nRuns with artifacts >= {min_mb} MB in Experiment {experiment_id}:")
        print(f"{'Run ID':<35} | {'Run Name':<30} | {'Size (MB)':<15}")
        print("-" * 85)
        for r in large_runs:
            print(f"{r['run_id']:<35} | {r['run_name']:<30} | {r['total_artifact_size_mb']:<15.2f}")
        print("-" * 85)
        
        return large_runs

    def delete_failed_runs(self, experiment_id: str, dry_run: bool = True) -> None:
        """Safely cleans up FAILED runs. CRITICAL: defaults to dry_run=True."""
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="attributes.status = 'FAILED'"
        )
        
        count = len(runs)
        if count == 0:
            print("No failed runs found to delete.")
            return

        if dry_run:
            print(f"DRY RUN: Found {count} failed runs that WOULD be deleted.")
            for run in runs:
                print(f"  - Would delete: {run.info.run_id}")
            print(f"Total runs that will be deleted if dry_run=False: {count}")
        else:
            print("WARNING: You have set dry_run=False. This will permanently delete data!")
            print(f"Deleting {count} failed runs...")
            for run in runs:
                self.client.delete_run(run.info.run_id)
            print(f"Successfully deleted {count} failed runs.")

    def export_run(self, run_id: str, output_directory: str) -> None:
        """Exports a run's configuration and artifacts to a local directory."""
        os.makedirs(output_directory, exist_ok=True)
        run = self.client.get_run(run_id)
        
        with open(os.path.join(output_directory, "params.json"), "w") as f:
            json.dump(run.data.params, f, indent=4)
            
        with open(os.path.join(output_directory, "metrics.json"), "w") as f:
            json.dump(run.data.metrics, f, indent=4)
            
        with open(os.path.join(output_directory, "tags.json"), "w") as f:
            json.dump(run.data.tags, f, indent=4)
            
        artifacts_dir = os.path.join(output_directory, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Download all associated artifacts locally
        artifact_uri = f"runs:/{run_id}/"
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=artifacts_dir)
        
        print(f"Exported run {run_id} to {output_directory}")

    def import_run(self, source_directory: str, experiment_id: str) -> str:
        """Restores a run's parameters, metrics, and tags into a new run."""
        with open(os.path.join(source_directory, "params.json"), "r") as f:
            params = json.load(f)
            
        with open(os.path.join(source_directory, "metrics.json"), "r") as f:
            metrics = json.load(f)
            
        with open(os.path.join(source_directory, "tags.json"), "r") as f:
            tags = json.load(f)
            
        run = self.client.create_run(experiment_id=experiment_id, tags=tags)
        new_run_id = run.info.run_id
        
        if params:
            for k, v in params.items():
                self.client.log_param(new_run_id, k, v)
                
        if metrics:
            for k, v in metrics.items():
                self.client.log_metric(new_run_id, k, v)
                
        return new_run_id

    def get_storage_report(self) -> None:
        """Iterates across all experiments and generates an artifact storage report."""
        experiments = self.client.search_experiments()
        
        print("\nStorage Report across all Experiments:")
        print(f"{'Experiment Name':<30} | {'Total Runs':<15} | {'Total Artifact Size (MB)':<25}")
        print("-" * 75)
        
        for exp in experiments:
            runs = self.client.search_runs([exp.experiment_id])
            num_runs = len(runs)
            
            total_size_bytes = 0
            for run in runs:
                total_size_bytes += self._get_run_artifact_size_bytes(run.info.run_id)
                
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            print(f"{exp.name:<30} | {num_runs:<15} | {total_size_mb:<25.2f}")
        print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="MLflow Artifact Manager CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Subcommand: report
    subparsers.add_parser("report", help="Print global storage report")
    
    # Subcommand: cleanup <experiment_id>
    cleanup_parser = subparsers.add_parser("cleanup", help="Safely cleanup failed runs")
    cleanup_parser.add_argument("experiment_id", type=str, help="Experiment ID to target")
    cleanup_parser.add_argument("--execute", action="store_true", help="DISABLE dry-run and permanently delete data")
    
    # Subcommand: export <run_id> <output_dir>
    export_parser = subparsers.add_parser("export", help="Export a run securely")
    export_parser.add_argument("run_id", type=str, help="Target run ID")
    export_parser.add_argument("output_dir", type=str, help="Target local directory")

    args = parser.parse_args()
    manager = ArtifactManager()
    
    if args.command == "report":
        manager.get_storage_report()
    elif args.command == "cleanup":
        # CRITICAL SAFETY: True unless explicit --execute flag is passed
        dry_run = not args.execute
        manager.delete_failed_runs(args.experiment_id, dry_run=dry_run)
    elif args.command == "export":
        manager.export_run(args.run_id, args.output_dir)

if __name__ == "__main__":
    main()
