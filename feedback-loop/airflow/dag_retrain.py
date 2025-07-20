from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

with DAG("retrain_on_drift", start_date=datetime(2025,1,1), schedule_interval="0 * * * *", catchup=False):
    check = BashOperator(task_id="run_drift_check", bash_command="python GithubActions/feedback-loop/monitor/drift_check.py")
    wait = FileSensor(task_id="wait_flag", filepath="/tmp/DRIFT_FLAG", path_interval=60, timeout=3600)
    retrain = BashOperator(task_id="retrain_model", bash_command="python GithubActions/feedback-loop/train_model.py && rm /tmp/DRIFT_FLAG")

    check >> wait >> retrain

