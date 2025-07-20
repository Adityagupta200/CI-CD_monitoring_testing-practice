import mlflow, pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset
import os

def load_data(n=500):
    client = mlflow.tracking.MLflowClient()
    runs = client.search_runs(experiemnt_ids=["1"], order_by=["metrics.ts DESC"], max_results=n)
    records = [r.data.params | r.data.metrics for r in runs]
    return pd.DataFrame(records)

ref = load_data(5000) # Earlier window
curr = load_data(500) # Latest
report = Report([DataDriftPreset()])

if not os.path.exists('/tmp'):
    os.makedirs('/tmp')

report.run(reference_data=ref, current_data=curr)
drift_score = report.as_dict()["metrics"][0]["result"]["dataset_drift"]

if drift_score["drift_detected"]:
    open("/tmp/DRIFT_FLAG", "w").close()
