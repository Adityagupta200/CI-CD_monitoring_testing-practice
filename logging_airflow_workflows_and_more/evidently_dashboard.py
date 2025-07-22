import pandas as pd


# Example dataframes; replace with your real data
ref_data = pd.read_csv("C:/Users/abc/Downloads/SWaT.A8_June 2021/For Mark/_20210624_100741/20210624_111312.csv")
cur_data = pd.read_csv("C:/Users/abc/Downloads/SWaT.A8_June 2021/For Mark/_20210624_100741/20210624_122008.csv")

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])

report.run(reference_data=ref_data, current_data=cur_data)

# Save as HTML file
report.save_html("evidently_drift_report.html")