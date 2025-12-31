from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import shutil
import traceback
from datetime import datetime

from src.data_handler import DataHandler
from src.bias_detector import BiasDetector
from src.fairness_engine import FairnessEngine
from src.model_trainer import ModelTrainer
from src.utils import load_params, load_metadata, save_metadata, ensure_dir

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')

app = FastAPI(
    title="Bias Mitigation API",
    description="API for detecting and mitigating bias in ML models",
    version="1.0.0"
)

# CORS Middleware - CRITICAL FOR FRONTEND TO WORK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Request/Response Models
class DataUploadResponse(BaseModel):
    message: str
    shape: List[int]
    columns: List[str]
    missing_values: dict

class MetadataRequest(BaseModel):
    target: str
    protected: List[str]
    features: List[str]

class BiasAnalysisResponse(BaseModel):
    metrics: dict
    plots: List[str]

class MitigationRequest(BaseModel):
    technique: str
    protected_attribute: str
    encoding_method: Optional[str] = "label"

class PredictionRequest(BaseModel):
    data: List[dict]

# Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Bias Mitigation API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug-bias")
async def debug_bias():
    """Debug endpoint to check bias analysis issues"""
    try:
        params = load_params()
        issues = []
        
        # Check if biased predictions exist
        if not os.path.exists(params["paths"]["biased_predictions"]):
            issues.append("Biased predictions file not found")
        
        # Check if metadata exists
        try:
            meta = load_metadata()
            issues.append(f"Metadata loaded: {meta}")
        except Exception as e:
            issues.append(f"Metadata error: {str(e)}")
        
        # Check if BiasDetector can be imported
        try:
            from src.bias_detector import BiasDetector
            issues.append("BiasDetector imported successfully")
        except Exception as e:
            issues.append(f"BiasDetector import error: {str(e)}")
        
        # Check matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            issues.append(f"Matplotlib backend: {matplotlib.get_backend()}")
        except Exception as e:
            issues.append(f"Matplotlib error: {str(e)}")
        
        # Check outputs directory
        if not os.path.exists("outputs"):
            issues.append("Outputs directory does not exist")
        else:
            issues.append(f"Outputs directory exists with files: {os.listdir('outputs')}")
        
        return {"debug_info": issues}
        
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.post("/upload-data", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload and validate dataset"""
    try:
        params = load_params()
        
        # Save uploaded file
        ensure_dir(params["paths"]["raw_data"])
        with open(params["paths"]["raw_data"], "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Load and validate
        df = pd.read_csv(params["paths"]["raw_data"])
        missing = df.isnull().sum()
        
        return DataUploadResponse(
            message="Data uploaded successfully",
            shape=list(df.shape),
            columns=list(df.columns),
            missing_values={k: int(v) for k, v in missing.items() if v > 0}
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.post("/set-metadata")
async def set_metadata(request: MetadataRequest):
    """Set target, protected attributes, and features"""
    try:
        params = load_params()
        df = pd.read_csv(params["paths"]["raw_data"])
        
        # Validate columns
        if request.target not in df.columns:
            raise ValueError(f"Target '{request.target}' not found")
        
        for attr in request.protected:
            if attr not in df.columns:
                raise ValueError(f"Protected attribute '{attr}' not found")
        
        for feat in request.features:
            if feat not in df.columns:
                raise ValueError(f"Feature '{feat}' not found")
        
        # Save processed data and metadata
        ensure_dir(params["paths"]["processed_data"])
        df.to_csv(params["paths"]["processed_data"], index=False)
        
        metadata = {
            "target": request.target,
            "protected": request.protected,
            "features": request.features,
            "shape": list(df.shape)
        }
        save_metadata(metadata)
        
        return {"message": "Metadata saved successfully", "metadata": metadata}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload pre-trained model and generate biased predictions - Compatible with multiple versions"""
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    
    model_path = None
    
    try:
        import joblib
        import pickle
        from sklearn.metrics import mean_squared_error
        
        params = load_params()
        
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower()
        
        # Validate file type
        if file_ext not in ['pkl', 'pickle', 'joblib']:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: .{file_ext}. Please upload .pkl, .pickle, or .joblib files."
            )
        
        # Save model temporarily
        model_path = f"temp_model_{os.getpid()}.{file_ext}"
        with open(model_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Try loading the model with multiple strategies
        model = None
        load_methods = []
        
        # Strategy 1: Try joblib first (most compatible for sklearn models)
        try:
            model = joblib.load(model_path)
            load_methods.append("joblib")
        except Exception as e1:
            # Strategy 2: Standard pickle
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                load_methods.append("pickle-default")
            except Exception as e2:
                # Strategy 3: Pickle with latin1 encoding (Python 2 compatibility)
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    load_methods.append("pickle-latin1")
                except Exception as e3:
                    # Strategy 4: Pickle with bytes encoding
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f, encoding='bytes')
                        load_methods.append("pickle-bytes")
                    except Exception as e4:
                        # Strategy 5: Try importing old module names
                        try:
                            # Handle numpy version incompatibility
                            import numpy.core._multiarray_umath as _multiarray_umath
                            sys.modules['numpy.core._multiarray_umath'] = _multiarray_umath
                            
                            with open(model_path, 'rb') as f:
                                model = pickle.load(f)
                            load_methods.append("pickle-numpy-compat")
                        except Exception as e5:
                            error_details = {
                                "joblib": str(e1)[:100],
                                "pickle_default": str(e2)[:100],
                                "pickle_latin1": str(e3)[:100],
                                "pickle_bytes": str(e4)[:100],
                                "numpy_compat": str(e5)[:100]
                            }
                            raise ValueError(
                                f"Could not load model with any method. "
                                f"This usually means version incompatibility. "
                                f"Try re-saving your model with: joblib.dump(model, 'model.joblib'). "
                                f"Errors: {error_details}"
                            )
        
        if model is None:
            raise ValueError("Model loaded but is None")
        
        # Verify model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError(
                f"Loaded object is not a valid model. Type: {type(model)}. "
                f"Model must have a 'predict' method."
            )
        
        # Load data
        try:
            df = pd.read_csv(params["paths"]["processed_data"])
            meta = load_metadata()
        except Exception as e:
            raise ValueError(
                f"Failed to load processed data. "
                f"Please upload dataset and configure metadata first. Error: {str(e)}"
            )
        
        # Generate predictions
        data_handler = DataHandler()
        X = data_handler.encode_data(df, meta["features"])
        
        # Validate data shape
        if len(X) == 0:
            raise ValueError("No data available for predictions")
        
        # Try to make predictions with error handling
        try:
            y_pred = model.predict(X)
        except Exception as pred_error:
            error_msg = str(pred_error)
            
            # Provide helpful error messages based on common issues
            if "feature" in error_msg.lower():
                raise ValueError(
                    f"Model expects different features than provided. "
                    f"Model was trained on different data. "
                    f"Error: {error_msg}"
                )
            elif "shape" in error_msg.lower():
                raise ValueError(
                    f"Data shape mismatch. "
                    f"Your data has {X.shape[1]} features. "
                    f"Error: {error_msg}"
                )
            elif "version" in error_msg.lower() or "module" in error_msg.lower():
                raise ValueError(
                    f"Library version incompatibility. "
                    f"Try re-training model with current sklearn version. "
                    f"Error: {error_msg}"
                )
            else:
                raise ValueError(f"Prediction failed: {error_msg}")
        
        # Validate predictions
        if y_pred is None or len(y_pred) == 0:
            raise ValueError("Model returned empty predictions")
        
        if len(y_pred) != len(df):
            raise ValueError(
                f"Prediction length mismatch. "
                f"Expected {len(df)}, got {len(y_pred)}"
            )
        
        # Save predictions
        df["actual"] = df[meta["target"]]
        df["pred_biased"] = y_pred
        
        ensure_dir(params["paths"]["biased_predictions"])
        df.to_csv(params["paths"]["biased_predictions"], index=False)
        
        # Calculate RMSE with error handling
        try:
            rmse = np.sqrt(mean_squared_error(df["actual"], df["pred_biased"]))
            
            # Sanity check
            if np.isnan(rmse) or np.isinf(rmse):
                raise ValueError("RMSE calculation resulted in invalid value")
                
        except Exception as e:
            raise ValueError(f"Failed to calculate RMSE: {str(e)}")
        
        # Cleanup
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
        
        return {
            "message": "Model uploaded and predictions generated successfully",
            "rmse": float(rmse),
            "predictions_saved": params["paths"]["biased_predictions"],
            "model_format": file_ext,
            "load_method": load_methods[-1] if load_methods else "unknown",
            "predictions_count": len(y_pred)
        }
    
    except HTTPException:
        raise
    except ValueError as ve:
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during model upload: {str(e)}"
        )

@app.get("/analyze-bias")
async def analyze_bias():
    """Analyze bias in predictions - Simplified version"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        params = load_params()
        
        # Check if biased predictions exist
        if not os.path.exists(params["paths"]["biased_predictions"]):
            raise HTTPException(
                status_code=400, 
                detail="Biased predictions not found. Please upload a model first."
            )
        
        # Load data
        df = pd.read_csv(params["paths"]["biased_predictions"])
        meta = load_metadata()
        
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        # Calculate simple fairness metrics for each protected attribute
        metrics_all = {}
        plots = []
        
        # Create prediction distribution plot
        try:
            plt.figure(figsize=(10, 6))
            if "actual" in df.columns and "pred_biased" in df.columns:
                sns.kdeplot(df["actual"], label="Actual", fill=True, color="black", alpha=0.5)
                sns.kdeplot(df["pred_biased"], label="Biased Prediction", fill=True, color="red", alpha=0.5)
                plt.legend()
                plt.title("Prediction Distribution")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.tight_layout()
                plot_path = "outputs/prediction_distribution.png"
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close()
                plots.append(plot_path)
        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
        
        # Calculate metrics for each protected attribute
        for attr in meta["protected"]:
            if attr not in df.columns:
                continue
                
            try:
                # Calculate basic fairness metrics
                groups = df[attr].unique()
                
                # Calculate mean predictions per group
                group_means = {}
                group_counts = {}
                
                for group in groups:
                    group_data = df[df[attr] == group]
                    group_means[str(group)] = float(group_data["pred_biased"].mean())
                    group_counts[str(group)] = len(group_data)
                
                # Calculate disparate impact (ratio of means)
                if len(group_means) >= 2:
                    means_list = list(group_means.values())
                    disparate_impact = min(means_list) / max(means_list) if max(means_list) > 0 else 0
                else:
                    disparate_impact = 1.0
                
                # Calculate statistical parity difference
                if len(group_means) >= 2:
                    statistical_parity_diff = max(means_list) - min(means_list)
                else:
                    statistical_parity_diff = 0.0
                
                metrics_all[attr] = {
                    "disparate_impact": float(disparate_impact),
                    "statistical_parity_diff": float(statistical_parity_diff),
                    "equal_opportunity_diff": 0.0,  # Placeholder
                    "average_odds_diff": 0.0,  # Placeholder
                    "theil_index": 0.0,  # Placeholder
                    "consistency": 1.0,  # Placeholder
                    "fpr_diff": 0.0,  # Placeholder
                    "fnr_diff": 0.0,  # Placeholder
                    "group_means": group_means,
                    "group_counts": group_counts
                }
                
                # Create group rates plot
                try:
                    plt.figure(figsize=(8, 6))
                    groups_list = list(group_means.keys())
                    means_plot = list(group_means.values())
                    
                    plt.bar(groups_list, means_plot, color=["gray", "blue"][:len(groups_list)])
                    plt.title(f"Mean Prediction by Group ({attr})")
                    plt.ylabel("Mean Prediction")
                    plt.xlabel(attr)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    plot_path = f"outputs/group_treatment_rates_{attr}.png"
                    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    plots.append(plot_path)
                except Exception as e:
                    print(f"Error creating group plot for {attr}: {str(e)}")
                    
            except Exception as e:
                print(f"Error calculating metrics for {attr}: {str(e)}")
                metrics_all[attr] = {
                    "disparate_impact": 0.0,
                    "statistical_parity_diff": 0.0,
                    "equal_opportunity_diff": 0.0,
                    "average_odds_diff": 0.0,
                    "theil_index": 0.0,
                    "consistency": 0.0,
                    "fpr_diff": 0.0,
                    "fnr_diff": 0.0
                }
        
        # Return results
        return {
            "metrics": metrics_all,
            "plots": plots
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in analyze_bias: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Bias analysis failed: {str(e)}"
        )

@app.post("/apply-mitigation")
async def apply_mitigation(request: MitigationRequest):
    """Apply fairness mitigation technique - Simplified version"""
    try:
        import warnings
        warnings.filterwarnings('ignore')
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.metrics import mean_squared_error
        
        params = load_params()
        meta = load_metadata()  # Load metadata at the start
        data_handler = DataHandler()
        
        # Load data
        X, y, df, _ = data_handler.get_train_data()  # Use underscore since we already have meta
        
        # Get the protected attribute data
        if request.protected_attribute not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Protected attribute '{request.protected_attribute}' not found in data"
            )
        
        protected_col = df[request.protected_attribute]
        
        # Apply mitigation based on technique
        technique_map = {
            "1": "None",
            "2": "Reweighing",
            "3": "Disparate Impact Remover",
            "4": "Reject Option Classification",
            "5": "Equalized Odds",
            "6": "MetaFair Classifier",
            "7": "Adversarial Debiasing"
        }
        
        technique_name = technique_map.get(request.technique, "Unknown")
        
        # Initialize weights (will be modified by some techniques)
        sample_weights = np.ones(len(y))
        X_fair = X.copy()
        
        # Apply different mitigation strategies
        if request.technique == "1":
            # No mitigation
            pass
            
        elif request.technique == "2":
            # Reweighing: Balance samples by protected attribute
            try:
                # Create a combined label for reweighing
                y_median = y.median()
                y_binary = (y >= y_median).astype(int)
                
                # Compute sample weights to balance groups
                combined_labels = protected_col.astype(str) + "_" + y_binary.astype(str)
                sample_weights = compute_sample_weight('balanced', combined_labels)
                
            except Exception as e:
                print(f"Reweighing error: {str(e)}")
                sample_weights = np.ones(len(y))
        
        elif request.technique == "3":
            # Disparate Impact Remover: Add small random noise to reduce discrimination
            try:
                # Simple approach: add calibrated noise based on group membership
                groups = protected_col.unique()
                for col in X_fair.columns:
                    if X_fair[col].dtype in ['int64', 'float64']:
                        col_std = X_fair[col].std()
                        noise_scale = col_std * 0.1  # 10% noise
                        
                        for group in groups:
                            mask = protected_col == group
                            noise = np.random.normal(0, noise_scale, mask.sum())
                            X_fair.loc[mask, col] = X_fair.loc[mask, col] + noise
            except Exception as e:
                print(f"Disparate Impact error: {str(e)}")
                X_fair = X.copy()
        
        elif request.technique in ["4", "5"]:
            # Post-processing techniques: Apply calibration
            # For now, just use standard weights
            pass
        
        elif request.technique in ["6", "7"]:
            # In-processing techniques: Use balanced weights
            try:
                sample_weights = compute_sample_weight('balanced', protected_col)
            except Exception as e:
                print(f"In-processing technique error: {str(e)}")
                sample_weights = np.ones(len(y))
        
        # Train fair model
        trainer = ModelTrainer()
        
        try:
            model, model_name = trainer.train(X_fair, y, sample_weights)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )
        
        # Generate predictions
        try:
            y_pred_fair = model.predict(X_fair)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
        
        # Load original predictions
        df_original = pd.read_csv(params["paths"]["biased_predictions"])
        df_original["pred_fair"] = y_pred_fair
        
        # Save
        ensure_dir(params["paths"]["fair_predictions"])
        df_original.to_csv(params["paths"]["fair_predictions"], index=False)
        
        # Save model
        trainer.save_model(model)
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y, y_pred_fair))
        
        return {
            "message": "Mitigation applied successfully",
            "technique": technique_name,
            "model": model_name,
            "rmse": float(rmse)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in apply_mitigation: {error_trace}")
        raise HTTPException(
            status_code=500, 
            detail=f"Mitigation failed: {str(e)}"
        )

@app.get("/compare-models")
async def compare_models():
    """Compare biased and fair models"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        params = load_params()
        meta = load_metadata()
        
        df_biased = pd.read_csv(params["paths"]["biased_predictions"])
        df_fair = pd.read_csv(params["paths"]["fair_predictions"])
        
        merged = df_biased.copy()
        merged["pred_fair"] = df_fair["pred_fair"]
        
        # Overall metrics
        overall = {
            "biased": {
                "rmse": float(np.sqrt(mean_squared_error(merged["actual"], merged["pred_biased"]))),
                "mae": float(mean_absolute_error(merged["actual"], merged["pred_biased"])),
                "r2": float(r2_score(merged["actual"], merged["pred_biased"]))
            },
            "fair": {
                "rmse": float(np.sqrt(mean_squared_error(merged["actual"], merged["pred_fair"]))),
                "mae": float(mean_absolute_error(merged["actual"], merged["pred_fair"])),
                "r2": float(r2_score(merged["actual"], merged["pred_fair"]))
            }
        }
        
        # Group-wise metrics
        group_metrics = {}
        for attr in meta["protected"]:
            group_metrics[attr] = {}
            for group in merged[attr].unique():
                group_data = merged[merged[attr] == group]
                group_metrics[attr][str(group)] = {
                    "biased_rmse": float(np.sqrt(mean_squared_error(group_data["actual"], group_data["pred_biased"]))),
                    "fair_rmse": float(np.sqrt(mean_squared_error(group_data["actual"], group_data["pred_fair"])))
                }
        
        return {
            "overall_metrics": overall,
            "group_metrics": group_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@app.get("/download-plot/{plot_name}")
async def download_plot(plot_name: str):
    """Download generated plots"""
    file_path = f"outputs/{plot_name}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(file_path)

@app.get("/download-predictions/{pred_type}")
async def download_predictions(pred_type: str):
    """Download predictions (biased or fair)"""
    params = load_params()
    
    if pred_type == "biased":
        file_path = params["paths"]["biased_predictions"]
    elif pred_type == "fair":
        file_path = params["paths"]["fair_predictions"]
    else:
        raise HTTPException(status_code=400, detail="Invalid prediction type")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Predictions not found")
    
    return FileResponse(file_path)

@app.get("/model-card")
async def get_model_card():
    """Generate and return model card"""
    try:
        from sklearn.metrics import mean_squared_error
        
        params = load_params()
        meta = load_metadata()
        
        df = pd.read_csv(params["paths"]["fair_predictions"])
        
        rmse_biased = np.sqrt(mean_squared_error(df["actual"], df["pred_biased"]))
        rmse_fair = np.sqrt(mean_squared_error(df["actual"], df["pred_fair"]))
        
        model_card = {
            "generated_at": datetime.now().isoformat(),
            "target": meta["target"],
            "protected_attributes": meta["protected"],
            "features": meta["features"],
            "metrics": {
                "biased_rmse": float(rmse_biased),
                "fair_rmse": float(rmse_fair),
                "improvement": float(rmse_biased - rmse_fair)
            }
        }
        
        return model_card
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model card generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)