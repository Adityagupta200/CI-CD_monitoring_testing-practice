import pandas as pd


# Example dataframes; replace with your real data
ref_data = pd.read_csv("ref_dataset")
cur_data = pd.read_csv("cur_dataset")

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])

report.run(reference_data=ref_data, current_data=cur_data)

# Save as HTML file
report.save_html("evidently_drift_report.html")
