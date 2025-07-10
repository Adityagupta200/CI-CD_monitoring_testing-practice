import pytest

def test_data_preprocessing():
    """Test data preprocessing function returns correct format"""
    raw_data = [1, 2, 3, 4, 5]
    processed = preprocess_data(raw_data)

    assert isinstance(processed, list)
    assert len(processed) == len(raw_data)
    assert all(isinstance(x, float) for x in processed)

def test_model_pipeline_integration():
    """Test that all pipeline components work together"""
    # Load sample data
    data = load_sample_data()

    # Test full pipeline
    preprocessed = preprocess_data(data)
    model = load_model()
    predictions = model.predict(preprocessed)

    # Verify integration works
    assert predictions is not None
    assert len(predictions) == len(data)

def test_model_training_completion():
    """Test that moedl trains to completion without errors"""
    model = MLModel()
    X_train, y_train = load_training_data()

    # Test training completes
    model.train(X_train, y_train)
    assert model.istrained == True
    assert model.weights is not None

def test_minimum_score_requirement():
    model = MLModel()
    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    # Train the model
    model.train(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)

    # Minimum score requirement
    min_accuracy = 0.90
    assert accuracy >= MIN_ACCURACY, f"Model accuracy: {accuracy} below minmium accuracy requirement: {min_accuracy}"

def test_data_schema_compliance():
    """Test data quality and data schema compliance"""
    data = load_data()
    expected_schema = ['col1', 'col2', 'col3', 'col4', 'col5','col6','col7']

    assert isinstance(data, dict)
    assert all(x in data.keys for x in expected_schema)
    assert all(len(data[col]) > 0 for col in expected_schema)

def test_data_sufficiency():
    """Test that dataset has sufficient samples for training"""
    data = load_data()
    min_samples = 1000
    assert len(data) >= min_samples, f"Insufficient samples for training: {len(data)}, required samples: {min_samples}"

@pytest.mark.parametrize("input_text, expected_category", [
    ("Natural language processing tutorial", "nlp"),
    ("Computer vision with deep learning", "computer-vision"),
    ("Machine learning operations guide", "mlops"),
    ("Data science fundamentals", "data-science")
])
def test_minimum_functionality(input_text, expected_category):
    """Test basic categorization functionality"""
    model = MLModel()
    prediction = model.predict(input_text)
    assert prediction == expected_category, f"Prediction : {prediction} doesn't matches expected category: {expected_category}"

@pytest.mark.parametrize("original, modified",[
    ("Machine learning is amazing", "ML is amazing"),  # Abbreviation
    ("Deep learning models", "deep learning models"),  # Case change
    ("AI will revolutionize tech", "AI will revolutionize technology") # Synonym
])
def test_invariance_transformations(original, modified):
    """Test that minor changes don't affect predictions"""
    model = load_trained_model()
    original_predictions = model.predict(original)
    modified_predictions = model.predict(modified) 

    assert original_predictions == modified_predictions

@pytest.mark.parameterize("positive_text, negative text",[
    ("This product is excellent", "This product is terrible"),
    ("I love this service", "I hate this service"),
    ("Outstanding quality", "Poor quality")
])
def test_directional_expectations(positive_text, negative_text):
    model = load_trained_model()
    positive_score = model.predict_sentiment(positive_text)
    negative_score = model.predict_sentiment(negative_text)

    assert positive_score > negative_score

def test_model_output_statistical_validation():
    """Test model output using statistical validation"""
    model = load_regression_model()
    test_input = [1.0, 2.0, 3.0]

    # Run prediction multiple times
    predictions = [model.predict(test_input) for _ in range(10)]
    mean_prediction = sum(predictions) / len(predictions)

    # Statistical validation with tolerance
    expected_range = (4.8, 5.2) # Expected prediction range
    assert expected_range[0] <= mean_prediction <= expected_range[1]

def test_prediction_confidence_intervals():
    """Test that model predictions fall within confidence intervals"""
    model = load_trained_model()
    data = load_public_dataset()

    assert isinstance(data, dict)
    predictions = [model.predict(x) for x in data]
    for pred in predictions:
        assert pred['lower_bound'] <= pred['prediction'] <= pred['upper_bound']

        interval_width = pred['upper_bound'] - pred['lower_bound']
        assert interval_width < 2.0 # Maximum acceptable uncertainity
