import pytest

@pytest.fixture(scope="session")
def resource():
    print("Setting up resource for a session")
    return "session_resource"

class TestA:
    def test_1(self, resource):
        pass
    def test_2(self, resource):
        pass

class TestB:
    def test_3(self, resource):
        pass