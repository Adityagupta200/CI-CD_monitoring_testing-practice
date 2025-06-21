import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def make_dataset():
    """Factory fixture that craetes custom datasets"""
    def _make_dataset(n_samples=100, n_features=5, random_state=42):
        """Generate a dataset with specific parameters"""
        np.random.seed(random_state)
        
        # Create features
        X = np.random.randn(n_samples, n_features)
        
        # Create target (simple linear relationship with noise)
        y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)
        
        # Convert to DataFrame
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df["target"] = y
        
        return df
    
    return _make_dataset

def test_model_with_different_datasets(make_dataset):
    """Test using multiple dataset variations"""
    # Small dataset
    small_df = make_dataset(n_samples=50)
    assert small_df.shape == (50, 6)

    # Large dataset with more features
    large_df = make_dataset(n_samples=200, n_features=10)
    assert large_df.shape == (200,11)