import pytest

@pytest.fixture(scope="module")
def config():
    print("Loading config")
    return {"debug": True}

def test_a(config):
    assert config["debug"]

def test_b(config):
    assert config["debug"]