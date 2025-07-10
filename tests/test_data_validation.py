import pytest
import pandas as pd
import numpy as np
from scipy import stats
import great_expectations as ge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

"""

SCHEMA VALIDATION TESTS

"""
@pytest.mark.data_validation
def test_data_schema_compliance(sample_dataset, data_schema):
    """
    
    Test that dataset complies with the expected schema
    
    """
    # assert all(expected_column in set(sample_dataset.columns) for expected_column in set(data_schema['columns'])), f"Schema Mismatch: {set(data_schema['columns']) - set(sample_dataset.columns)}"

    # Check column names
    expected_columns = set(data_schema['columns'])
    actual_columns = set(sample_dataset.columns)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    assert len(missing_columns) == 0, f"Missing columns: {missing_columns}"
    assert len(extra_columns) == 0, f"Unexpected columns: {extra_columns}"

    print(f"Schema Validation passed: {len(actual_columns)} match expected_schema")

@pytest.mark.data_validation
def test_data_types_validation(sample_dataset, data_schema):
    expected_types = data_schema['dtypes']
    
    for column, expected_dtype in expected_types.items():
        actual_dtype = str(sample_dataset[column].dtype)

        # Handle different representations of the same type
        if expected_dtype in [int64, int32] and actual_dtype in [int64, int32]:
            continue
        elif expected_dtype in [float64, float32] and actual_dtype in [float64, float32]:
            continue
        else:
            assert actual_dtype == expected_type, f"Data type mismatch: {sample_dataset[column]} has type: {sample_dataset[column].dtype}, expected type: {expected_dtype}"
    print(f"Data type validation complete, {expected_types} match with sample dataset data types.")

@pytest.mark.data_validation
def test_required_columns_not_null(sample_dataset, data_schema):
    """
    
    Test that required columns don't have null values
    
    """
    required_columns = data_schema['required_columns']

    for req_cols in required_columns:
        null_values = sample_dataset[req_cols].isnull().sum()
        null_percentage = null_count/len(sample_dataset)
        max_allowed = data_schema['max_missing_percentage']
        assert null_values <= max_allowed, f"Null values detected in {req_cols} in sample dataset with null percentage to be: {null_percentage:.3f * 100.000}%, null values exceed the maximum allows: {max_allowed}" 
    print(f"Required columns validation passed: All within {data_schema['max_missing_percentage']*100}% missing threshold")

"""

DATA RANGE AND BOUNDARY VALIDATION

"""
@pytest.mark.data_validation
def test_feature_value_ranges(sample_dataset, data_schema):
    """
    
    Test that feature values are within expected ranges
    
    """
    expected_range = data_schema['ranges']

    for column, (min_val, max_val) in expected_range.items():
        if column in sample_dataset.columns:
            # Remove null values for range checking
            non_null_values = sample_dataset[column].dropna()

            actual_min = non_null_values.min()
            actual_max = non_null_values.max()

            assert actual_min >= min_val, f"Sample dataset values out of range, expected range: ({min_val, max_val})"
            assert actual_max <= max_val, f"Sample dataset values out of range, expected range: ({min_val, max_val})"

            print(f"Values of {column} in the sample dataset within expected ranges: ({actual_min, actual_max}) with the expected range being: ({min_val, max_val})")

    print(f"Test for feature value ranges passed.")

@pytest.mark.data_validation
def test_target_variable_distribution(sample_dataset):
    """
    
    Test target variable distribution for classification problem
    
    """
    target_column = "target"
    target_values = sample_dataset[target_column].value_counts()

    # Test that we have both classes
    unique_classes = set(target_values.index)
    expected_classes = {0, 1}

    assert unique_classes == expected_classes, f"Expected classes: {expected_classes}, got: {unique_classes}"
    
    # Test class imbalance isn't too extreme (minority class >= 5%)
    min_class_percentage = target_values.min() / len(sample_dataset)
    assert min_class_percentage >= 0.05, \
        f"Minority class percentage {min_class_percentage:.3f} too low, potential severe imbalance"
    print(f"Target distribution validation passed: {dict(target_values)}")


"""

STATISTICAL DATA VALIDATION

"""

@pytest.mark.data_validation
def test_feature_distribution(sample_dataset):
    """
    
    Test that feature distributions are reasonable using statistical tests.

    """
    numerical_columns = ['feature_1', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    for column in numerical_columns:
        values = sample_dataset[column].dropna()

        # Test for extreme outliers using IQR method
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR

        extreme_outliers = ((values < lower_bound) | (values > upper_bound))
        outlier_percentage = extreme_outliers / len(values)

        assert outlier_percentage <= 0.05, f"Column {column} has outlier percentage: {outlier_percentage:.3f * 100.000}%, exceeds 5% threshold"

        # Test for normality
        try:
            _, p_value = stats.normaltest(values)
            normality_status = "normal" if p_value > 0.05 else "non_normal"
            
            print(f"{column}: {outlier_percentage:.3f * 100.000}% outliers, distribution is {normality_status}")
        except:
            print(f"{column}: {outlier_percentage:.3f * 100.000}% outliers")
    print(f"Feature Distribution Test passed")

@pytest.mark.data_validation
def test_feature_correlations(sample_dataset):
    """
    
    Test feature correlations to detect potential multicollinearity issues.

    """
    numerical_columns = ['feature_1', 'feature_2', "feature_3", 'featuer_4']
    correlation_matrix = sample_dataset[numerical_columns].corr()

    # Check for high correlations (excluding diagonal)
    high_correlation_threshold = 0.9

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            correlation = abs(correlation_matrix.iloc[i, j])

            assert correlation < high_correlation_threshold, f"High correlation {correlation:.3f} between '{col1}' and '{col2}' exceeds threshold {high_correlation_threshold}"
    print(f"Feature correlation validation passed: no highly correlated features")

"""

DATA QUALITY VALIDATION

"""
@pytest.mark.data_validation
def test_duplicate_records(sample_dataset):
    """
    
    Test for duplicate records in the dataset

    """
    total_records = len(sample_dataset)
    unique_records = len(sample_dataset.drop_duplicates())

    duplicate_records = total_records - unique_records
    duplicate_percentage = len(duplicate_records)/len(sample_dataset)

    assert duplicate_percentage <= 0.500, f"Number of duplicates found: {len(duplicate_records)}, also exceeding the threshold value of: 0.500, with the duplicate percentage being: {duplicate_percentage:.3f * 100.000}%"
    print(f"Test for finding duplicate records passed, with result being : No duplicates found")

@pytest.mark.data_validation
def test_data_completeness(sample_dataset):
    """
    
    Test overall data completeness across all columns
    
    """
    num_of_missing_values_across_each_column = sample_dataset.isnull.sum()
    missing_percentage = (num_of_missing_values_across_each_column.sum())/(len(sample_dataset) * len(sample_dataset.columns))
    completeness = 1 - missing_percentage

    assert completeness > 0.900, f"Many missing values found, with completeness percentage being: {completeness:.3f * 100.000}%"
    print(f"Data completeness test passed with completeness percentage being: {completeness:.3f * 100.000}%")

    # Print per column missing data summary
    # for column in sample_dataset.columns:
    #     num_of_missing_values = sample_dataset[column].isnull().sum()
    #     missing_percentage = num_of_missing_values/len(sample_dataset[column])
    #     completeness = 1 - missing_percentage

    #     print(f"Completeness percentage for column: {colums} is: {completeness:.3f * 100.000}% with number of missing values being: {num_of_missing_values}")
    for column_name, missing_count in num_of_missing_values_across_each_column.items():
        if missing_count > 0:
            missing_percentage = missing_count / len(sample_dataset)
            print(f"{column_name} has missing percentage: {missing_percentage:.3f * 100.000}% with number of missing values being: {missing_count}")

"""

CORRUPTED DATA VALIDATION

"""
@pytest.mark.data_validation
def test_corrupted_data_detection(corrupted_dataset, data_schema):
    """
    
    Test detection of various data corruption issues.
    
    """
    # Test missing value detection
    missing_values_for_each_column = corrupted_dataset.isnull().sum()
    assert missing_values_for_each_column.sum() == 0, f"Number of missing values in in each column: {missing_values_for_each_column}"

    # Test outlier detection in feature_2
    feature_2_values = corrupted_dataset['feature_2'].dropna()
    outliers = (feature_2_values > 100).sum() # Values we artificially set to 1000
    assert outliers > 0, "Should detect outliers in feature_2"

    # Test invalid data type detection in feature_3

    # feature_3_values = corrupted_dataset['feature_3'].dropna()
    # expected_data_type = data_schema['dtypes']['feature_3']
    # actual_data_type = corrupted_dataset['feature_3'].dtype

    # assert actual_data_type == expected_data_type, f"Unexpected data type found in feature_3 of value: {actual_data_type}, expected data type: {expected_data_type}"

    # Test invalid data type detection in feature_3
    try:
        pd.to_numeric(corrupted_dataset['feature_3'], errors='raise')
        pytest.fail("Should have detected non-numeric values in 'feature_3'")
    except:
        pass

    print(f"Corrupted data detection test passed with number of missing values in each column being: {missing_values_for_each_column}\n")
    print(f"Number of outliers in feature_2 in the corrupted dataset to be: {outliers}\n")
    print("Invalid data types correctly detected")

"""

GREAT EXPECTATIONS INTEGRATION

"""
@pytest.mark.data_validation
def test_great_expectations_validation(sample_dataset):
    """
    
    Demonstrate using Great Expectations for advanced data validation

    """
    # Convert pandas dataframe to Great Expectations Dataframe

    ge_df = ge.from_pandas(sample_dataset)

    expectations = [
        # Column existence expectations
        ge_df.expect_column_to_exist('feature_1'),
        ge_df.expect_column_to_exist('feature_2'),
        ge_df.expect_column_to_exist('target'),

        # Value expectations
        ge_df.expect_column_values_not_to_be_between('target', min_value=0,max_value=1),
        ge_df.expect_column_values_to_not_be_null('target'),

        # Statistical expectations
        ge_df.expect_column_mean_to_be_between('feature_1', min_value=-2, max_value=2),
        ge_df.expect_column_stdev_to_be_between('feature_1', min_value=0.5, max_value=2),

        # Uniqueness expectations
        ge_df.expect_column_values_to_be_unqiue('target', mostly=0.3) # At least 30% unique
    ]
    
    # Validate all expectations
    failed_expectations = []
    for expectation in expectations:
        if not expectation.success:
            failed_expectations.append(expectation)
    
    assert len(failed_expectations) == 0, \
        f"Failed expectations: {[exp.expectation_config.expectation_type for exp in failed_expectations]}"
    print(f"Great Expectations validation passed: {len(expectations)} expectations met")

"""

DATA DRIFT DETECTION

"""
@pytest.mark.data_validation
def test_data_drift_detection(sample_dataset, test_data_generator):
    """
    
    Test for data drift by comparing distributions between reference and new data
    
    """
    # Use original data as reference
    reference_data = sample_dataset

    # Generate new data (simulating production data)
    new_X, new_y = test_data_generator(n_samples=500)
    new_data = pd.DataFrame({
        'feature_1': new_X[:,0],
        'feature_2': new_X[:,1],
        'feature_3': new_X[:,2],
        'feature_4': new_X[:,3],
        'target': new_y
    })

    # Perform Kolmogorov-Smrinov test for distribution drift
    drift_threshold = 0.05 # p-value threshold

    for column in ['feature_1', 'feature_2', 'feature_3', 'feature_4']:
        ref_values = reference_data[column].dropna()
        new_values = new_data[column].dropna()

        # Perform KS test
        ks_statistic, p_value = stats.ks_2samp(ref_values, new_values)

        # Check for significant drift
        drift_detected = p_value < drift_threshold

        if drift_detected:
            warnings.warn(f"Data drift detected in column: {column}: p-value: {p_value:.4f}"
            f"drift= {'Yes'}")

"""

FEATURE ENGINEERING VALIDATION

"""
@pytest.mark.data_validation
def test_feature_scaling_validation(preprocessed_data):
    """
    
    Test that feature scaling was applied correctly.
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data

    # Test that scaled features have approximately zero mean and unit variances
    for i in range(X_train.shape[1]):
        feature_mean = np.mean(X_train[:,i])
        feature_std = np.std(X_train[:,i])

        assert abs(feature_mean) < 0.1, f"Feature {i} mean {feature_mean:.4f} not close to zero after scaling"
        assert abs(feature_std - 1.0) < 0.1, f"Feature {i} std {feature_std:.4f} not close to 1.0 after scaling"
    print("Feature scaling validation passed: all features properly normalized")

@pytest.mark.data_validation
def test_train_test_split_validation(preprocessed_data, ml_pipeline_config):
    """
    
    Test that train-test split was performed correctly
    
    """
    X_train, X_test, y_train, y_test, scaler = preprocessed_data
    expected_test_size = ml_pipeline_config['test_size']

    total_samples = len(X_train) + len(X_test)
    actual_test_percentage = len(X_test) / total_samples

    assert abs(actual_test_percentage - expected_test_size) < 0.02, f"Unequal test dataset size: {actual_test_percentage}, expected test size: {expected_test_size}"

    # Test that split preserved class distribution (stratification)
    train_class_split = np.bincount(y_train)/len(y_train)
    test_class_split = np.bincount(y_test)/len(y_test) 

    for i in range(len(train_class_split)):
        dist_diff = abs(train_class_split[i] - test_class_split[test_class_split])
        assert dist_diff < 0.1, f"Differnece between the train and test class distributions vary beyond the threshold value of 0.1 with the distribution differnece being: {dist_diff}."
    print(f"Train-test split validation passed: {actual_test_size:.3f} test size, stratified")

"""

PARAMETERIZED DATA VALIDATION

"""
@pytest.mark.data_validation
@pytest.mark.parametrize("column", "expected_type",[
    ("feature_1", -5, 5),
    ("feature_2", 0, 12),
    ("feature_3", 0, 10),
    ("feature_4", 0, 20),
    ("target", 0, 1)
])
def test_parametrize_column_types(sample_dataset, column, expected_type):
    """
    
    Parameterized test for validating individual column data types.
    
    """
    actual_type = sample_dataset[column].dtype
    assert actual_type == expected_type, f"Unexpected data type found in sample dataset: {actual_type}, expected data type: {expected_type}"

@pytest.mark.data_validation
@pytest.mark.parametrize(column, min_val, max_val, [
    ("feature_1", -5, 5),
    ("feature_2", 0, 12),
    ("feature_3", 0, 10),
    ("feature_4", 0, 20),
    ("target", 0, 1)
])
def test_parameterized_value_ranges(sample_dataset, column, min_val, max_val):
    """
    
    Test that values are within ranges defined in the above parameterize marker.
    
    """
    non_null_values = samples_dataset[column].dropna()
    actual_min_value = non_null_values.min()
    actual_max_value = non_null_values.max()

    assert actual_max_value <= max_val, f"Values out of bound, {column} has value: {acutal_max_value} exceeding the expected maximum value {max_val}"
    assert actual_min_value >= min_val, f"Values out of bound, {column} has value: {actual_min_value} less than the expected minimum value {min_val}"

    print(f"Parameterized test for testing the value ranges of different numeric columns in the sample dataset passed!")
