import pytest
import os
import shutil

@pytest.fixture
def temp_model_artifacts():
    #Setup: Create a folder
    os.makedirs("./temp_models", exist_ok=True)
    yield "./temp_models" # Give the path to the text
    #Teardown: remove the folder and everything in it
    shutil.rmtree("./temp_models")

def test_save_model(temp_model_artifacts):
    path = temp_model_artifacts
    # Save a file in the folder
    with open(os.path.join(path, "model.txt"), "w") as f:
        f.write("model data")
    assert os.path.exists(os.path.join(path, "model.txt"))