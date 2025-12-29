import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from src.utils import load_params, ensure_dir, setup_mlflow, log_metrics, log_params

class ModelTrainer:
    def __init__(self):
        self.params = load_params()
        
    def train(self, X, y, sample_weights=None):
        """Train best model with hyperparameter tuning"""
        print("🧠 Training model...")
        
        # Detect task type
        is_classification = self._is_classification(y)
        task = "Classification" if is_classification else "Regression"
        print(f"📋 Task: {task}")
        
        # Get models and params
        models, param_grids, scoring, cv = self._get_models_and_params(is_classification)
        
        # Train and select best
        best_model = None
        best_score = -np.inf if is_classification else np.inf
        best_name = ""
        
        mlflow = setup_mlflow()
        
        for name, model in models.items():
            print(f"\n🔧 Training {name}...")
            
            try:
                search = RandomizedSearchCV(
                    model, param_grids[name],
                    scoring=scoring, cv=cv,
                    n_iter=min(5, max(1, len(param_grids[name]))),
                    random_state=42, n_jobs=-1
                )
                
                if sample_weights is not None:
                    search.fit(X, y, sample_weight=sample_weights)
                else:
                    search.fit(X, y)
                
                score = search.best_score_
                display_score = score if is_classification else -score
                metric = "Accuracy" if is_classification else "RMSE"
                print(f"✅ {name} {metric}: {display_score:.4f}")
                
                # Track with MLflow
                with mlflow.start_run(run_name=f"train_{name}", nested=True):
                    log_params({f"{name}_{k}": v for k, v in search.best_params_.items()})
                    log_metrics({f"{name}_{metric}": display_score})
                
                is_better = score > best_score if is_classification else -score < best_score
                if is_better:
                    best_score = score
                    best_model = search.best_estimator_
                    best_name = name
                    
            except Exception as e:
                print(f"⚠️ {name} failed: {e}")
        
        final_metric = "Accuracy" if is_classification else "RMSE"
        final_score = best_score if is_classification else -best_score
        print(f"\n🏆 Best: {best_name} ({final_metric} = {final_score:.4f})")
        
        return best_model, best_name
    
    def save_model(self, model):
        """Save trained model"""
        ensure_dir(self.params["paths"]["model"])
        joblib.dump(model, self.params["paths"]["model"])
        print(f"✅ Model saved to {self.params['paths']['model']}")
    
    def load_model(self):
        """Load trained model"""
        return joblib.load(self.params["paths"]["model"])
    
    def _is_classification(self, y):
        """Detect if task is classification"""
        unique_vals = np.unique(y)
        return np.array_equal(unique_vals, [0, 1]) or (
            len(unique_vals) <= 5 and np.issubdtype(y.dtype, np.integer)
        )
    
    def _get_models_and_params(self, is_classification):
        """Get models and hyperparameters based on task"""
        if is_classification:
            models = {
                "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
                "DecisionTree": DecisionTreeClassifier(random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear')
            }
            
            param_grids = {
                "RandomForest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [5, 10, None],
                    "min_samples_split": [2, 5]
                },
                "DecisionTree": {
                    "max_depth": [3, 5, 10, None],
                    "min_samples_split": [2, 5]
                },
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1, 10]
                }
            }
            
            scoring = "accuracy"
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            models = {
                "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "GradientBoosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, verbosity=0),
                "DecisionTree": DecisionTreeRegressor(random_state=42),
                "LinearRegression": LinearRegression()
            }
            
            param_grids = {
                "RandomForest": {
                    "n_estimators": [50, 100, 150],
                    "max_depth": [5, 10, None]
                },
                "GradientBoosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                },
                "DecisionTree": {
                    "max_depth": [3, 5, 10, None]
                },
                "LinearRegression": {}
            }
            
            scoring = "neg_root_mean_squared_error"
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        return models, param_grids, scoring, cv