import mlflow
import mlflow.sklearn

def start_experiment(exp_name="Customer_Churn_Experiments"):
    mlflow.set_experiment(exp_name)

def log_model_and_metrics(model, params, metrics):
    # Log parameters
    for k, v in params.items():
        mlflow.log_param(k, v)

    # Log metrics
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")
