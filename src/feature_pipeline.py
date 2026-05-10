"""
feature_pipeline.py

Reusable, versioned feature engineering module for the Anti-Gravity project.
Applies transformations consistently across training, retraining, and serving.
"""

import os
import hashlib
import joblib
import numpy as np
import pandas as pd
from typing import List, Union

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

import mlflow

class FeaturePipeline:
    """
    Feature engineering pipeline that applies standard scaling, interaction features,
    variance thresholding, and correlation-based feature selection.
    """
    def __init__(self):
        self.pipeline = None

    def fit_transform(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Fits the transformation pipeline on training data ONLY.
        Applies: StandardScaler -> interaction features (degree=2) -> 
                 remove low-variance (threshold=0.01) -> select top 20 by correlation.
        Returns transformed numpy array.
        """
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('variance_filter', VarianceThreshold(threshold=0.01)),
            ('selector', SelectKBest(score_func=f_classif, k=20))
        ])
        
        # Fit and transform on training data only to prevent data leakage
        X_transformed = self.pipeline.fit_transform(X_train, y_train)
        
        # Save the fitted pipeline using joblib
        local_path = "feature_pipeline.pkl"
        joblib.dump(self.pipeline, local_path)
        
        # Compute SHA-256 hash
        with open(local_path, "rb") as f:
            pipeline_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Log to MLflow if an active run exists
        active_run = mlflow.active_run()
        if active_run:
            mlflow.log_artifact(local_path)
            mlflow.set_tag("feature_pipeline_hash", pipeline_hash)
            
        return X_transformed

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Applies the ALREADY FITTED pipeline to new data.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        return self.pipeline.transform(X)

    def get_feature_names(self) -> List[str]:
        """
        Returns a list of the feature names that were selected (after the selection step).
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first.")
        
        try:
            return list(self.pipeline.get_feature_names_out())
        except Exception:
            # Fallback if names are not easily obtainable
            return [f"feature_{i}" for i in range(20)]

    @classmethod
    def load_pipeline(cls, run_id: str) -> "FeaturePipeline":
        """
        Downloads feature_pipeline.pkl from the MLflow run with that run_id,
        loads and returns the pipeline instance using joblib.
        """
        artifact_uri = f"runs:/{run_id}/feature_pipeline.pkl"
        local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
        
        loaded_sklearn_pipeline = joblib.load(local_path)
        
        instance = cls()
        instance.pipeline = loaded_sklearn_pipeline
        return instance
