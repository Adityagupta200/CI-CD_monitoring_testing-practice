# import pytest

# @pytest.fixture
# def demo_fixture(request):
#     print("Setup for test")
#     def cleanup1():
#         print("Cleanup for step 1")
#     def cleanup2():
#         print("Cleanup for step 2")
#     request.addfinalizer(cleanup1)
#     request.addfinalizer(cleanup2)
#     return "resource"

# def test1(demo_fixture):
#     assert demo_fixture == "resource"

import pytest
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture
def trained_model(request):
    model = RandomForestClassifier(n_estimators=10)
    model.fit([[0], [1]], [0,1])
    def teardown():
        print("Releasing model resources")
    request.addfinalizer(teardown)
    return model

def test_model(trained_model):
    assert hasattr(trained_model, "predict")