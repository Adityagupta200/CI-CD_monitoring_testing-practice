import mlflow, joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("model-train")
with mlflow.start_run():
    model = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    mlflow.log_metric("val_acc", acc)
    joblib.dump(model, "model.pkl")