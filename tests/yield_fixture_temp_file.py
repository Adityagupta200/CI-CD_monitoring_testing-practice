import pytest
import os

@pytest.fixture
def temp_file():
    file_path = "test_data.txt"
    with open(file_path, "w") as f:
        f.write("test data")
    yield file_path
    os.remove(file_path)


def test_1(temp_file):
    pass