import yaml
import json
import os
import mlflow
import dagshub

def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_metadata():
    """Load metadata from JSON"""
    params = load_params()
    with open(params["paths"]["metadata"], "r") as f:
        return json.load(f)

def save_metadata(meta):
    """Save metadata to JSON"""
    params = load_params()
    os.makedirs(os.path.dirname(params["paths"]["metadata"]), exist_ok=True)
    with open(params["paths"]["metadata"], "w") as f:
        json.dump(meta, f, indent=2)

def setup_mlflow():
    """Setup MLflow with DagsHub"""
    params = load_params()
    if params["dagshub"]["mlflow_tracking"]:
        dagshub.init(
            repo_owner=params["dagshub"]["repo_owner"],
            repo_name=params["dagshub"]["repo_name"],
            mlflow=True
        )
    return mlflow

def log_metrics(metrics, step=None):
    """Log metrics to MLflow"""
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)

def log_params(params_dict):
    """Log parameters to MLflow"""
    mlflow.log_params(params_dict)

def ensure_dir(filepath):
    """Ensure directory exists"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)