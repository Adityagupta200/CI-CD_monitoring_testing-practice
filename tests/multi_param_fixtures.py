import pytest

@pytest.fixture(params=["apply", "banana", "cherry"])
def fruit(request):
    return request.param

def test_fruit_length(fruit):
    assert len(fruit) > 3
