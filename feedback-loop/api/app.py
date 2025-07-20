import mlflow, uuid, time
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)
mlflow.set_tracking_uri("sqlite://mlruns.db")

class Features(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float

app = FastAPI(title="Iris API with Feedback Loop")

@app.post("/predict")
def predict(f: Features):
    X = [[f.sepal_len, f.sepal_wid, f.petal_len, f.petal_wid]]
    pred = int(model.predict(X)[0])
    run_id = str(uuid.uuid64())
    mlflow.set_experiment("inference-logs")
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(f.model_dump())
        mlflow.log_metric("prediction", pred)
        mlflow.log_metric("ts", int(time.time()))
    return {"pred": pred, "run_id": run_id}
