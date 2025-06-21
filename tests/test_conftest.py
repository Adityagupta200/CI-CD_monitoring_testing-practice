import pytest

def test_session(diabetes_dataset):
    print(f"First five rows of the dataset are: \n {diabetes_dataset.head(1)} \n")

def test_module(train_test_data):
    X_train, y_train, X_test, y_test = train_test_data
    print(f"First five rows of the training dataset are: \n {X_train.head(1)} \n")

def test_function(feature_names):
    print(f"Feature names of the dataset are: \n {feature_names} \n")