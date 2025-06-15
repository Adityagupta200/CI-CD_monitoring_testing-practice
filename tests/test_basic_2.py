def greet(name):
    print("Hello, " + name)

def test_greet():
    assert greet("Alice") == "Hello, Alice"