# import time
# import pytest

# def test_model_resposnse_time():
#     model = load_trained_model()
#     data = load_data()
    
#     current_time = time.time()
#     predictions = model.predict(data)
#     response_time = time.time() - current_time

#     max_response_time = 0.1
#     assert response_time < max_response_time, f"Response time too much: {response_time:.3f}"

# @pytest.mark.parameterize("batch_size", [1, 10, 100, 1000])
# def test_model_scalability(batch_size):
#     model = load_model()
#     test_data = generate_batch_data(batch_size)

#     start_time = time.time()
#     predictions = model.predict(test_data)
#     end_time = time.time()

#     assert len(predictions) == batch_size

#     # Test that processing time scales reasonably
#     processing_time = end_time - start_time
#     time_per_sample = processing_time / batch_size
#     max_response_time = 0.1
#     assert time_per_sample < max_response_time, f"Response took too long and exceeding max_response_time limit: {processing_time}"

# def test_data_availability():
#     """Test that requires data sources are available"""
#     data_sources = ['training_data.csv', 'validation_data.csv', 'test_data.csv']
#     for source in data_sources:
#         assert os.path.exists(source), f"Data source {source} not found"
# def test_data_correctness():
#     """Test data has correct sizing and no missing values"""
#     data = load_data()

#     assert len(data) > 0, "Dataset is empty"

#     assert data.isnull().sum().sum() == 0, "Dataset contains Null values"

#     assert data['feature1'].dtype == np.float64, "featuers should be floating poitn numbers"
#     assert data['target'].dtype = np.int64, "Target should be integer"








