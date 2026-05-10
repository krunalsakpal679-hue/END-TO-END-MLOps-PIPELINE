"""
test_feature_pipeline.py

Unit tests for the feature engineering pipeline.
"""

import os
import sys
import pytest
import numpy as np
import mlflow
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# We import from src directly to satisfy IDE linters like Pylance
from src.feature_pipeline import FeaturePipeline

@pytest.fixture
def synthetic_data():
    """Generates synthetic dataset for testing."""
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_fit_transform_shape(synthetic_data):
    """Checks that the output has the right shape (n_samples, <=20)."""
    X_train, _, y_train, _ = synthetic_data
    pipeline = FeaturePipeline()
    
    with mlflow.start_run():
        X_out = pipeline.fit_transform(X_train, y_train)
        
    assert X_out.shape[0] == X_train.shape[0]
    assert X_out.shape[1] <= 20

def test_no_leakage(synthetic_data):
    """
    Fits on X_train, transforms X_test, checks no information from X_test 
    was used in fitting by asserting batch vs row-by-row transforms match.
    """
    X_train, X_test, y_train, _ = synthetic_data
    pipeline = FeaturePipeline()
    
    with mlflow.start_run():
        pipeline.fit_transform(X_train, y_train)
        
    X_test_batch_out = pipeline.transform(X_test)
    X_test_row_out = pipeline.transform(X_test[0:1])
    
    # If standard scaling utilized the whole X_test batch to compute mean/std, 
    # the outputs would diverge when scaled individually.
    np.testing.assert_allclose(X_test_batch_out[0], X_test_row_out[0], rtol=1e-5)

def test_transform_before_fit_raises(synthetic_data):
    """Checks that calling transform before fit_transform raises RuntimeError."""
    X_train, _, _, _ = synthetic_data
    pipeline = FeaturePipeline()
    
    with pytest.raises(RuntimeError, match="Pipeline not fitted. Call fit_transform first."):
        pipeline.transform(X_train)

def test_round_trip(synthetic_data):
    """Save then load the pipeline, check predictions are identical."""
    X_train, _, y_train, _ = synthetic_data
    pipeline = FeaturePipeline()
    
    # Create an MLflow run to log the artifact
    with mlflow.start_run() as run:
        X_out_original = pipeline.fit_transform(X_train, y_train)
        run_id = run.info.run_id
        
    # Load the pipeline using the class method pointing to the run_id
    loaded_pipeline = FeaturePipeline.load_pipeline(run_id)
    X_out_loaded = loaded_pipeline.transform(X_train)
    
    # The output from the freshly loaded pipeline should match perfectly
    np.testing.assert_array_equal(X_out_original, X_out_loaded)
