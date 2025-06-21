# src/calculator.py
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

def power(a, b):
    return a ** b

def complex_function(x, y, z):
    # This function has multiple branches
    if x > 0:
        if y > 0:
            return x + y + z
        else:
            return x + z
    else:
        if z > 0:
            return y + z
        else:
            return x