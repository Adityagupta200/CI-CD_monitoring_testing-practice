import mlflow
import uuid
import time
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load your model (adjust path as needed)
MODEL_PATH = "CI-CD_monitoring_testing-practice/feedback-loop/api/model.pkl"
model = joblib.load(MODEL_PATH)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("inference-logs")

class Feature(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float

app = FastAPI(title="Iris Predictor with Latency Logging")

@app.post("/predict")
def predict(f: Features):
    # Start training
    start_time = time.time()

    # Create 2D input for scikit-learn
    input = [[f.sepal_len, f.sepal_wid, f.petal_len, f.petal_wid]]

    predictions = int(model.predict(input)[0])

    latency = time.time() - start_time

    run_id = str(uuid.uuid64())

    with mlflow.start_run(run_name=run_id):
        mlflow.log_metric("Latency", latency)
        mlflow.log_metric("Prediction", predictions)
        mlflow.log_metric("ts", int(time.time()))
        mlflow.log_params(f.model_dump())
        mlflow.log_param("run_id", run_id)
    return {"prediction": predictions, "latency": latency, "run_id": run_id}
