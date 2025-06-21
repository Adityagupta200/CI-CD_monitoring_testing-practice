import pytest
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


# @pytest.fixture(scope="package")
# def db_connection():
#     print("Connecting to DB")
#     return "db_conn"

@pytest.fixture(scope="session")
def diabetes_dataset():
    """Load diabetes dataset once for entire test session"""
    return load_diabetes(return_X_y=False, as_frame=True).frame

@pytest.fixture(scope="module")
def train_test_data(diabetes_dataset):
    """Create train/test split to be shared across test module"""
    X = diabetes_dataset.drop("target", axis=1)
    y = diabetes_dataset["target"]
    return train_test_split(X, y, test_size=0.25, random_state=42)

@pytest.fixture
def feature_names(diabetes_dataset):
    """Provide feature names from the dataset"""
    return diabetes_dataset.columns.drop("target").tolist()
