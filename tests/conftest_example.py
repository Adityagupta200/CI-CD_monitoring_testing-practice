import pytest
import numpy as np

@pytest.fixture
def sample_training_data():
    """Fixture providing consistent training data for tests"""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)

@pytest.fixture
def trained_model(sample_training_data):
    """Fixture providing a pre-trained model"""
    X, y = sample_training_data
    model = MLModel()
    model.train(X, y)
    return model

@pytest.fixture
def model_performance_threshold():
    """Fixture defining minimum performance requirements"""
    return {
        'accuracy: 0.85',
        'precision: 0.80',
        'recall: 0.80',
        'f1_score: 0.80'
    }