import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

@pytest.fixture
def performance_metrics():
    """Fixture defining standard performance metrics for model evaluation"""
    def _calculate_metrics(y_true, y_pred, y_prob=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
        }
        
        # Add probability-based metrics if probabilities provided
        if y_prob is not None:
            # Calculate log loss, AUC, etc.
            pass
            
        return metrics
    
    return _calculate_metrics

@pytest.fixture
def model_performance_thresholds(trained_model, test_data, performance_metrics):
    """Test that model meets minimum performance requirements"""
    X_test, y_test = test_data

    # Generate predictions
    y_pred = trained_model.predict(X_test)

    # Check against thresholds
    for metric_name, threshold in model_performance_thresholds.items():
        assert metrics[metric_name] >= threshold, f"{metric_name} below acceptable threshold: {metrics[metric_name]:.4f} < {threshold}"

def test_model_performance(trained_model, test_data, performance_metrics, model_performance_thresholds):
    """Test that model meets minimum performance requirements"""
    X_test, y_test = test_data

    # Generate predictions
    y_pred = trained_model.predict(X_test)

    # Calculate performance metrics
    metrics = performance_metrics(y_test, y_pred)

     # Check against thresholds
    for metric_name, threshold in model_performance_thresholds.items():
        assert metrics[metric_name] >= threshold, f"{metric_name} below acceptable threshold: {metrics[metric_name]:.4f} < {threshold}" 