# Example: Testing a feature engineering function

# import pytest
import numpy as np
# from src.features import normalize_data

def test_normalize_data():
  # Arrange
  input_data = np.array([1, 2, 3, 4, 5])
  expected_output = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

  # Act
  result = normalize_data(input_data)

  # Assert
  np.testing.assert_array_almost_equal(result, expected_output)
#   result_test = np.testing.assert_array_almost_equal(result, expected_output)
#   return result_test


def normalize_data(data):
  return data / np.max(data)

def test_full_pipeline_integration():
    # Test that data preprocessing -> Feature engineering -> model training works
    raw_data = load_test_data()
    processed_data = preprocess_data(raw_data)
    features = extract_features(processed_data)
    model = train_model(features)

    assert model is not None
    assert model.score(features) > 0.8

def test_data_scheme_validation():
    df = load_training_data()
    
    # Check required columns exist
    required_columns = ['feature1', 'feature2', 'target']
    assert all(col in df.columns for col in required_columns)
    
    # Check data types
    assert df['feature1'].dtype == 'float64'
    assert df['target'].dtype == 'int64'

    # Check for missing values
    assert df.isnull().sum().sum() == 0

def test_model_accuracy_threshold():
    model = load_trained_model()
    test_data, test_labels = load_test_dataset()
    predic

def __main__():
  test_normalize_data()

if __name__ == "__main__":
    __main__()