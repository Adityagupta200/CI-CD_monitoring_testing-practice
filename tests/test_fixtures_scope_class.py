import pytest

@pytest.fixture(scope="class")
def db_connection():
    print("Connecting to DB")
    return "db_conn"

class TestDB1:
    def test_query1(self, db_connection):
        assert db_connection == "db_conn"
    
    def test_query2(self, db_connection):
        assert db_connection == "db_conn"

class TestDB2:
    def test_query3(self, db_connection):
        assert db_connection == "db_conn"
    
    def test_query4(self, db_connection):
        assert db_connection == "db_conn"