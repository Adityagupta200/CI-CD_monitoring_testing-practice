import pytest
import pandas as pd
import numpy as np
from src.models import load_model
from src.data_validation import validate_data_schema, check_data_drift
from src.bias_detection import detect_bias
from src.explainability import generate_explanations

class ModelQualityGates:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.test_data = pd.read_csv(test_data_path)

    def test_data_quality_gate(self):
        """Ensure input data meets quality standards"""
        # Schema Validation
        schema_valid = validate_data_schema(self.test_data)
        assert schema_valid, "Data schema validation failed"

        # Check for data drift
        drift_detected = check_data_drift(self.test_data)
        assert not drift_detected, "Significant data drift detected"

        # Missing value check
        missing_percentage = self.test_data.isnull().sum().sum()/len(self.test_data)
        assert missing_percentage < 0.05, "High missing percentage"

    def test_bias_detection_gate(self):
        """Ensure model doesn't exhibit unfair bias"""
        protected_attributes = ['gender', 'age-group', 'ethnicity']

        for attribute in protected_attributes:
            if attribute in self.test_data.columns:
                bias_score = detect_bias(self.model, self.test_data, attribute)
                assert bias_score < 0.1, f"Bias detected for {attribute}: {bias_score:.3f}"
    
    def test_explainability_gate(self):
        """Ensure model predictions are explainable"""
        sample_data = self.test_data.sample(n=10)
        explanations = generate_explanations(self.model, sample_data)

        # Check that explanations exist for all samples
        assert len(explanations) == 10, "Explanations for all samples don't exist"

        # Check explanation quality (feature importance coverage)
        for explanation in explanations:
            total_importance = sum(explanation['feature_importance'].values())
            assert total_importance > 0.8, "Explanation coverage too low"
        
    def test_response_time_gate(self):
        """Ensure model meets latency requirements"""
        import time
        sample_input = self.test_data.iloc[0:1]

        # Measure prediction time
        start_time = time.time()
        prediction = self.model.predict(sample_input)
        end_time = time.time()

        # Check response time
        response_time = end_time - start_time
        assert response_time < 0.1, f"Response too slow: {response_time:.3f}s "
    
    def test_resource_utilization_gate(self):
        """Ensure model doesn't exceed resource limits"""
        import psutil
        import os

        # Get current process
        process = psutil.Process(os.getpid())

        # Memory usage check 
        memory_usage = process.memory_ingo.rss / 1024 * 1024
        assert memory_usage < 100, f"Memory usage too high: {memory_usage: .1f}"

        # CPU usage check during prediction
        cpu_before = process.cpu_percent()
        predictions = self.model.predict(self.test_data.sample(n=100))
        cpu_after = process.cpu_percent()

        # Check CPU usage 
        cpu_usage = cpu_after - cpu_before
        assert cpu_usage < 10, f"CPU usage too high: {cpu_usage:.1f}"

# PyTest implementation
@pytest.fixture
def quality_gates():
    return ModelQualityGates('model/latest_model.pkl', 'data/test_data.csv')

def test_data_quality_gate(quality_gates):
    quality_gates.test_data_quality_gate()

def test_bias_detection_gate(quality_gates):
    quality_gates.test_bias_detection_gate()

def test_explainability_gate(quality_gates):
    quality_gates.test_explainability_gate()

def test_response_time_gate(quality_gates):
    quality_gates.test_response_time_gate()

def test_resource_utilization_gate(quality_gates):
    quality_gates.test_resource_utilization_gate
