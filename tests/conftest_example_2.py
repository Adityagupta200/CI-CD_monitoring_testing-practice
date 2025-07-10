import pytest
import pandas as pd
import numpy as np
import time 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tempfile
import os

"""Test fixtures and utilities"""

@pytest.fixture(scope="session")
def sample_dataset():
    """Session-scoped fixture providing consistent sample dataset across all tests.
    Returns:
        pd.DataFrame: Sample dataset with features and target"""
    
    np.random.seed(42)
    data = {
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.uniform(0, 10, 1000),
        'feature_4': np.random.exponential(2, 1000),
        'target': np.random.choice([0, 1], 1000, p=[0.6, 0.4])
    }
    return pd.DataFrame(data)

@pytest.fixture(scope="function")
def preprocessed_data(sample_dataset):
    """Function-scoped fixture providing preprocessed data for each test
    Args:
        sample_dataset: Sample dataset from session fixture"""
    X = sample_dataset.drop('target', axis=1)
    y = sample_dataset['target']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

@pytest.fixture(scope="module")
def trained_models(preprocessed_data):
    """Module-scoped fixture providing trained models to performance testing
    Args:
        preprocessed_data: Preprocessed training data
    Returns:
        dict: Dictionary of trained models"""
    
    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    models = {}

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    models['random_forest'] = rf_model

    # Train the Logistic Regresstion Model
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    models['lr_model'] = lr_model

    return models

@pytest.fixture
def corrupted_dataset(sample_dataset):
    """
    Fixture providing dataset with various data quality issues for testing.

    Args: 
        sample_dataset: CLean sample dataset

    Returns:
        pd.DataFrame: Dataset with introduced data quality issues
    """

    corrupted_data = sample_dataset.copy()

    # Introduce missing values
    corrupted_data.loc[0:10, 'feature_1'] = np.nan

    # Introduce outliers
    corrupted_data.loc[20:25, 'feature_2'] = 1000

    # Introduce wrong data types
    corrupted_data.loc[30:35, 'feature_3'] = 'invalid_value'

    return corrupted_data

@pytest.fixture
def temp_model_path():
    """
    Fixture providing temporary file path for model serialization tests.

    Yields:
        str: Temporary file path
    """
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        yield tmp_file.name
    
    # CLeanup after test
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)

def test_model_save_and_load(temp_model_path):
    model = RandomForestClassifier() # Create a model
    save_model(model, temp_model_path) # Save the model

    loaded_model = load_model(temp_model_path)
    assert loaded_model == model
    
"""
PERFORMANCE TESTING FIXTURES
"""
@pytest.fixture
def perfomance_data_generator():
    """
    Fixture for generating data of vaious sizes for performance testing.

    Returns:
        function: Data generator function
    """
    def generate_data(n_samples=1000, n_features=4):
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        return X, y
    return generate_data
    
@pytest.fixture
def benchmark_config():
    """
    Configuration for performance benchmarks.

    Returns:
        dict: Benchmark configuration parameters.
    """
    return {
        'max_inference_time': 0.1, # Maximum allowed inference time in seconds
        'max_training_time': 5.0, # Maximum allowed training time in seconds
        'memory_limit_mb': 500, # Memory limit in MB
        'min_accuracy': 0.7, # Minimum required accuracy
        'data_sizes': [100, 500, 1000, 5000], # Different data sizes to test
    }

"""

DATA VALIDATION FIXTURES

"""

@pytest.fixture
def data_schema():
    """
    
    Expected data schema for validation tests

    Returns:
        dict: Expected data schema

    """
    return {
        'columns': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target'],
        'dtypes': {
            'feature_1': 'float64',
            'feature_2': 'float64', 
            'feature_3': 'float64',
            'feature_4': 'float64',
            'target': 'int64'
        },
        'ranges': {
            'feature_1': (-5, 5),
            'feature_2': (0, 12),
            'feature_3': (0, 10),
            'feature_4': (0, 20),
            'target': (0, 1)
        },
        'required_columns': ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target'],
        'max_missing_percentage': 0.05  # Maximum 5% missing values allowed
    }

@pytest.fixture
def ml_pipeline_config():
    """
    
    Configuration for ML pipeline testing
    
    """
    return {
        'test_size': 0.2,
        'random_state': 42,
        'cv_folds': 3,
        'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1'],
        'feature_selection_threshold': 0.01,
        'scaling_method': 'standard'
    }

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    
    config.addinivalue_line(
        "markers", "data_validation: mark test as data validation test"
    )
    
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )

    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )

def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names"""
    for item in items:
        # Mark performance tests
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        # Mark data validation tests
        if "data_validation" in item.name or "schema" in item.name:
            item.add_marker(pytest.mark.data_validation)

        # Mark slow tests
        if "slow" in item.name or "integration" in item.name:
            item.add_marker(pytest.mark.slow)


"""

Utility Functions

"""
        
class TestDataGenerator:
    """
    
    Utility class for generating test data with specific characteristics
    
    """
    @staticmethod
    def create_imbalanced_dataset(n_samples=1000, imbalance_ratio=0.1):
        """Create dataset with class imbalance"""
        n_minority = int(n_samples * imbalance_ratio)
        n_majority = n_samples - n_minority

        X_minority = np.random.randn(n_minority, 4) + 2
        X_majority = np.random.randn(n_majority, 4)

        X = np.vstack([X_minority, X_majority])
        y = np.hstack([np.ones(n_minority), np.zeros(n_majority)])

        return X, y
    
    @staticmethod
    def create_high_dimensional_dataset(n_samples=1000, n_features=100):
        """Create high-dimensional dataset for testing."""
        X = np.random.randn(n_samples, n_features)
        y = np.random.choice([0, 1], n_samples)
        return X, y

    @pytest.fixture
    def test_data_generator():
        """Fixture providing TestDataGenerator utility"""
        return TestDataGenerator()

    """
    
    COMMAND LINE OPTIONS
    
    """

    def pytest_addoption(parser):
        """Add custom command line options."""
        parser.addoption(
            "--dataset-path",
            action="store",
            default=None,
            help="Path to dataset for testing"
        )
        
        parser.addoption(
            "--model-path",
            action="store",
            default=None,
            help="Path to pre-trained model for testing"
        )

        parser.addoption(
            "--performance-threshold",
            action="store",
            default=0.1,
            help="Performance threshold in seconds"
        )

@pytest.fixture
def dataset_path(request):
    """Fixture for dataset path from command line."""
    return request.config.getoption("--dataset-path")

@pytest.fixture
def model_path(request):
    """Fixture for model path from command line"""
    return request.config.getoption("--model-path")

@pytest.fixture
def performance_threshold(request):
    return request.config.getoption("--performance-threshold")