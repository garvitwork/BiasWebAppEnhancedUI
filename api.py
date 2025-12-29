from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

from src.data_handler import DataHandler
from src.bias_detector import BiasDetector
from src.fairness_engine import FairnessEngine
from src.model_trainer import ModelTrainer
from src.utils import load_params, load_metadata, save_metadata, ensure_dir

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

@app.get("/analyze-bias", response_model=BiasAnalysisResponse)
async def analyze_bias():
    """Analyze bias in predictions"""
    try:
        detector = BiasDetector()
        metrics = detector.analyze()
        
        # List generated plots
        plots = [
            "outputs/prediction_distribution.png",
        ] + [f"outputs/group_treatment_rates_{attr}.png" for attr in load_metadata()["protected"]]
        
        return BiasAnalysisResponse(
            metrics=metrics,
            plots=[p for p in plots if os.path.exists(p)]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias analysis failed: {str(e)}")

@app.post("/apply-mitigation")
async def apply_mitigation(request: MitigationRequest):
    """Apply fairness mitigation technique"""
    try:
        engine = FairnessEngine()
        data_handler = DataHandler()
        
        # Load data
        X, y, df, meta = data_handler.get_train_data()
        y_bin = (y >= y.median()).astype(int)
        
        # Apply mitigation
        result = engine._apply_technique(
            request.technique,
            X, y_bin,
            request.protected_attribute,
            df
        )
        
        if not result:
            raise ValueError("Mitigation technique failed")
        
        X_fair, y_fair, weights = result
        
        # Train fair model
        trainer = ModelTrainer()
        model, model_name = trainer.train(X_fair, y, weights)
        
        # Generate predictions
        df["pred_fair"] = model.predict(X_fair)
        
        # Save
        params = load_params()
        ensure_dir(params["paths"]["fair_predictions"])
        df.to_csv(params["paths"]["fair_predictions"], index=False)
        trainer.save_model(model)
        
        from sklearn.metrics import mean_squared_error
        rmse = np.sqrt(mean_squared_error(y, df["pred_fair"]))
        
        return {
            "message": "Mitigation applied successfully",
            "technique": engine._get_method_name(request.technique),
            "model": model_name,
            "rmse": float(rmse)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation failed: {str(e)}")

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