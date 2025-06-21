import pytest
import pandas as pd

@pytest.fixture
def sample_training_data():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'target': [0, 1, 0, 1, 0]
    })

def test_data_validation(sample_training_data):
    assert sample_training_data.shape[0] > 0
    assert sample_training_data.shape[1] == 3
    assert not sample_training_data.isnull().any().any()
