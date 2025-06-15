def greet(name):
    return("Hello, " + name)

def test_greet():
    assert greet("Alice") == "Hello, Alice"