import mlflow
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
model=joblib.load("models/xgb_model.pkl")
X_test=joblib.load("models/X_test.pkl")
y_test=joblib.load("models/y_test.pkl")
y_pred=model.predict(X_test)

rmse=np.sqrt(mean_squared_error(y_test,y_pred))

mae=mean_absolute_error(y_test,y_pred)

r2=r2_score(y_test,y_pred)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run():
    mlflow.log_param("model",model)


    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    mlflow.log_artifact("plots/shap_summary.png")
    mlflow.log_artifact("plots/shap_dependence_plot.png")
    mlflow.log_artifact("plots/waterfall_plot.png")

print("run logged successfully")