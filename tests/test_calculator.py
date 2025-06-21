import pytest
from src.calculator import add, subtract, multiply, divide, power, complex_function

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0

def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(2, 3) == -1

def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(-2, 3) == -6

def test_divide():
    assert divide(6, 3) == 2
    assert divide(5, 2) == 2.5

    with pytest.raises(ValueError):
        divide(1, 0)

def test_complex_funcion_partial():
    assert complex_function(1, 1, 1) == 3
