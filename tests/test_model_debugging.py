# File: test_model_debugging.py
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

def test_model_training_with_output():
    """Test model training with debug output"""
    print("Starting model training test...")
    
    # Generate sample data
    X = np.random.rand(100, 4)
    y = np.random.randint(0, 2, 100)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X, y)
    
    print(f"Model coefficients shape: {model.coef_.shape}")
    print(f"Model intercept: {model.intercept_}")
    
    # Test model properties
    assert model.coef_.shape[1] == 4, "Model should have 4 features"
    assert len(model.classes_) == 2, "Model should have 2 classes"
    
    print("Model training test completed successfully!")

def test_data_validation_with_prints():
    """Test data validation with debug prints"""
    data = {
        'train_size': 1000,
        'test_size': 200,
        'features': 10
    }
    
    print(f"Validating dataset configuration: {data}")
    
    assert data['train_size'] > data['test_size'], "Training set should be larger than test set"
    assert data['features'] > 0, "Should have at least one feature"
    
    print("Data validation passed!")
