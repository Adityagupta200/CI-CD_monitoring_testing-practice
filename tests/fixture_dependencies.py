import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

@pytest.fixture
def preprocessing_pipeline(feature_names):
    """Create a preprocessing pipeline using feature names"""
    return {
        "scaler": StandardScaler(),
        "feature_selector": SelectKBest(f_classif, k = min(5, len(feature_names)))
    }

@pytest.fixture
def processing_training_data(train_test_data, preprocessing_pipeline):
    """Apply preprocessing to training data"""
    (X_train, X_test, y_train, y_test) = train_test_data

    # Apply preprocessing
    scaler = preprocessing_pipeline["scaler"]
    feature_selector = preprocessing_pipeline["feature_selector"]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_selected = feature_selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = feature_selector.transform(X_test_scaled)
    
    return {
        "X_train": X_train_selected,
        "X_test": X_test_selected,
        "y_train": y_train,
        "y_test": y_test,
        "transformers": {
            "scaler": scaler,
            "feature_selector": feature_selector
        }
    }

def test_1(preprocessing_pipeline):
    pass