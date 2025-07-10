import pytest
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.metric import accuracy_score, precision_score, recall_score
import threading
import concurrent.futures
from memory_profiler import memory_usage

"""

BASIC PERFORMANCE TESTS

"""
@pytest.mark.performance
def test_model_inference_time(trained_models, preprocessed_data, performance_threshold):
    """
    
    Test that model inference time is within acceptable limits.

    This is a critical test for production deployments where response time matters.
    
    """

    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    for model_name, model in trained_models.items():
        # Warm up the model (first prediction is often slower)
        _=model.predict(X_test[:1])

        # Measure inference time for single prediction
        start_time = time.time()
        predictions = model.predict(X_test[:1])
        inference_time = time.time() - start_time

        print(f"\n {model_name} inference time: {inference_time:.4f}s")

        # Assert inference time is below threshold
        assert inference_time <= performance_threshold, f"Inference took too long for model: {model_name}, {inference_time}s"

@pytest.mark.performance
def test_batch_prediction_performance(trained_models, preprocessed_data, benchmark_config):
    """
    Test batch prediction performance with different batch sizes
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data
    batch_sizes = [1, 10, 50, 100, len(X_test)]

    for model_name, model in trained_model.items():
        performance_results = {}

        for batch_size in batch_sizes:
            if batch_size > len(X_test):
                batch_size = len(X_test)
            
            # Measure batch prediction time
            start_time = time.time()
            predictions = model.predict(X_test[:batch_size])
            batch_inference_time = time.time() - start_time()

            # Calculate throughput(predictions per second)
            throughput = batch_size/batch_time if batch_time > 0 else float('inf')
            performance_results[batch_size] = {
                'time': batch_time,
                'throughput': throughput
            }

            print(f"\n{model_name} - Batch size {batch_size}: {batch_time:.4f}s" f"Throughput: {thorughput:.2f}pred/sec")

        # Verify that larger batches are more efficient(higher throughput)
        throughputs = [performance_results[bs]['throughput'] for bs in batch_sizes if bs in performance_results]
        for i in range(1, len(throughputs)):
            assert throughputs[i] >= throughputs[i-1], "Throughputs should not decrease as batch size increases"
            

            
"""

SCALABILITY TESTING

"""
@pytest.mark.performance
@pytest.mark.slow
def test_model_scalability(trained_models, performance_data_generator, bechmark_config):
    """
    Test model performance across different data sizes to verify scalability.
    
    """    
    data_sizes = benchmark_cofig['data_sizes']

    for model_name, model in trained_models.items():
        scalability_results = {}

        for data_size in data_sizes:
            # Generate test data of specified size
            X_test, y_test = performance_data_generator(n_samples=data_size)

            # Measure prediction time
            start_time = time.time()
            predictions = model.predict(X_test)
            prediction_time = time.time() - start_time

            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            predictions_per_second = int(data_size/prediction_time)

            scalability_results[data_size] = {
                'time': prediction_time,
                'accuracy': accuracy,
                'predictions_per_second or throughput': predictions_per_second
            }
            print(f"Perfomance results of {model_name} with test data size- {data_size}:\n{scalability_results[data_size]}")
            assert accuracy >= benchmark_config['min_accuracy'], f"Accuracy of {model_name}: {accuracy:.4f} below threshold: {benchmark_config['min_accuracy']}"

        # Verify prediction time scales reasonably (not exponentially)
        times = [scalability_results[size]['time'] for size in data_sizes]

        size_ratios = [data_sizes[i] / data_sizes[0] for i in range(len(data_sizes))]

        time_ratios = [times[i] / times[0] for i in range(len(times))]

        # Time ratios should not increase more than quadratically with data sizes
        for i, (size_ratio, time_ratio) in enumerate(zip(size_ratios, time_ratios)):
            if size_ratio > 1:
                assert time_ratio <= size_ratio ** 2, \
                    f"{model_name} prediction time scaling is too poor: " \
                    f"size ratio {size_ratio:.2f}"

@pytest.mark.performance
def test_memory_usage_during_prediction(trained_models, preprocessed_data, benchmark_config):
    """
    Test memory usage during model prediction to ensure it stays within limits.
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data
    memory_limit_mb = benchmark_config['memory_limit_mb']

    for model_name, model in trained_models.items():
        
        def prediction_task():
            """Task to monitor memory usage during prediction."""
            return model.predict(X_test)

            # Monitor memory usage during prediction
            memory_before = psutil.Process.memory_info().rss / 1024 / 1024 # MB
            try:
                # Use memory_profiler to track peak memory usage
                mem_usage = memory_usage((prediction_task, ()), interval=0.01, timeout = 30)
                peak_memory = max(mem_usage) if mem_usage else memory_before

                memory_used = peak_memory - memory_before

                print(f"\n{model_name} memory usage : {memory_used:.2f} MB"
                f"Peak: {peak_memory:.2f} MB")

                # Assert memory usage is within limits
                assert memory_used <= memory_limit_mb,\
                    f"{model_name} exceeded memory limits, max memory available: {memory_limit_mb:.2f} MB"
            except ImportError:
                # Fallback if memory_profiler is not available
                predictions = moedl.predict(X_test)
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = memory_after - memory_before
                print(f"\n{model_name} memory usage(estimated): {memory_used:.2f}")
                assert memory_used <= memory_limit_mb, \
                    f"{model_name} estimated memory usage {memory_used:.2f} MB exceeds limit {memory_limit_mb} MB"

"""

CONCURRENT PERFORMANCE TESTING

"""
@pytest.mark.performance
def test_concurrent_predictions(trained_models, preprocessed_data):
    """

    Test model performance under concurrent load to simulate multiple users.

    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data
    num_threads = 5
    predictions_per_thread = 10

    for model_name, model in trained_models.items():
        def make_predictions(thread_id):
            """Function to run predictions in a seperate thread"""
            thread_results = []
            for i in range(predictions_per_thread):
                start_time = time.time()
                predictions = model.predict(X_test[i:i+1])
                prediction_time = time.time() - start_time
                thread_results.append(prediction_time)
            return thread_results
        
        # Run concurrent predictions
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_thread = {
                executor.submit(make_predictions, i): i
                for i in range(num_threads)
            }

            all_times = []
            for future in concurrent.futures.as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    thread_times = future.result()
                    all_times.extend(thread_times)
                except Exception as exc:
                    pytest.fail(f"Thread {thread_id} generated an exception: {exc}")
        total_time = time.time() - start_time
        avg_prediction_time = np.mean(all_times)

        print(f"{model_name} concurrent performance:")
        print(f" Total time: {total_time:.4f}s")
        print(f"Average prediction time: {avg_prediction_time:.4f}s")
        print(f" Total predictions: {len(all_times)}")
        print(f"Throughput: {len(all_times)/total_time:.2f}pred/sec")

        # Assert that concurrent performance is reasonable
        assert avg_prediction_time < 1.0,\
            f"{model_name} average prediction time: {avg_prediction_time:.4f}s too slow under concurrent load"


"""

STATISTICAL PERFORMANCE VALIDATION

"""
@pytest.mark.performance
def test_prediction_consistency(trained_models, preprocessed_data):
    """
    
    Test that a model performs well consistently over multiple runs
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data
    num_runs = 10

    for model_name, model in trained_models.items():
        all_predictions = []

        # Run prediction multiple times
        for run in range(num_runs):
            predictions = model.predict(X_test)
            all_predictions.extend(predictions)
        
        # Check consistency across runs
        for i in range(len(all_predictions)):
            consistency = np.array_equal(all_predictions[0], all_predictions[i])
            assert consistency, f"{model_name} has inconsistent predictions between 0 and {i}"
        print(f"{model_name} has consistent predictions across {num_runs} runs")


@pytest.mark.performance
def test_performance_regression(trained_models, preprocessed_data):
    """
    
    Test for performance regression by comparing against baseline metrics.

    In real scenarios, you would load baseline metrics from a file or database.
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    # Baseline performance metrics (would normally be loaded from storage)
    baseline_metrics = {
        'random_forest': {'accuracy': 0.80, 'inference_time': 0.01},
        'logistic_regression': {'accuracy': 0.75, 'inference_time': 0.005}
    }
    
    tolerance = 0.05 # 5% tolerance for regression detection

    for model_name, model in trained_models.items():
        if model_name not in baseline_metrics:
            pytest.skip(f"No baseline metrics available for {model_name}")
        baseline = baseline_metrics[model_name]

        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time
        accuracy = accuracy_score(y_test, predictions)

        print(f"\n{model_name} performance:")
        print(f"Current accuracy:{accuracy:.4f} (baseline: {baseline['accuracy']:.4f})")
        print(f"Current inference time: {inference_time:.4f}s (baseline: {baseline['inference_time']:.4f})")

        # Check for accuracy regression
        accuracy_regression = (baseline['accuracy'] - accuracy)/baseline['accuracy']
        assert accuracy_regression <= tolerance, f"{model_name} accuracy regression exceeds tolerance of {tolerance} with accuracy regression being equal to: {accuracy_regression}"

        # Check for performance regression (inference time should not increase significantly)
        time_regression = (inference_time - baseline['inference_time'])/baseline['inference_time']
        assert time_regression <= tolerance, f"{model_name} performance regression(inference-time regression) exceeds tolerance: {tolerance} with inference time equal to {inference_time}"

"""

BENCHMARK TESTING WITH PYTEST-BENCHMARK

"""
@pytest.mark.performance
def test_model_training_benchmark(benchmark, preprocessed_data):
    """
    
    Benchmark model training time using pytest-benchmark.

    Note: This requires pytest-benchmark to be installed: pip install pytest-benchmark
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    from sklearn.ensemble import RandomForestClassifier

    def train_model():
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    """Benchmark the training process"""
    model = benchmark(train_model)

    # Verify the model was trained correctly
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.5, f"Model's accuracy: {accuracy:.4f} too low"

@pytest.mark.performance
def test_prediction_performance(benchmark, preprocessed_data, trained_models, benchmark_config):
    """
    
    Benchmark prediction time for different models
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    for model_name, model in trained_models.items():
        # Make inference from each model and store inference time
        start_time = time.time()
        predictions = benchmark.pedantic(
            model.predict(X_test),
            rounds = 10,
            iterations = 3,
            warmup_rounds = 2)
        inference_time = time.time() - start_time
        assert inference_time > benchmark_config['max_inference_time'], f"Inference time too long for {model_name}: {inference_time:.4f}s"
        num_of_predictions = len(predictions)
        assert num_of_predictions == len(X_test), f"{model_name} has incorrect number predictions: {num_of_predictions}"
        assert all(pred in [0, 1] for pred in predictions), f"Predictions out of range of [0, 1] for {model_name}"
    
