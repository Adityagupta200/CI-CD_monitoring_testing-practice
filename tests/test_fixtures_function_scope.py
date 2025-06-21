import pytest

@pytest.fixture(scope="function")
def sample_list():
    print("Creating list")
    return []

def test_one(sample_list):
    sample_list.append(1)
    assert sample_list == [1]

def test_two(sample_list):
    assert sample_list == []
    