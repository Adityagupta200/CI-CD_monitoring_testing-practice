# File: test_mlops_pipeline.py
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def test_data_preprocessing():
    """Test data preprocessing pipeline"""
    # Create sample data
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })
    
    # Test data shape
    assert data.shape[0] > 0, "Dataset should not be empty"
    assert data.shape[1] == 3, "Dataset should have 3 columns"
    
    # Test for missing values
    assert not data.isnull().any().any(), "Dataset should not contain missing values"

def test_model_training():
    """Test model training process"""
    # Generate sample data
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Test model is trained
    assert hasattr(model, 'estimators_'), "Model should be fitted"
    assert len(model.estimators_) == 10, "Model should have 10 estimators"

def test_model_prediction():
    """Test model prediction functionality"""
    # Simple model setup
    X = [[1, 2], [3, 4], [5, 6], [7, 8]]
    y = [0, 1, 0, 1]
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict([[2, 3]])
    assert len(predictions) == 1, "Should return one prediction"
    assert predictions[0] in [0, 1], "Prediction should be 0 or 1"
