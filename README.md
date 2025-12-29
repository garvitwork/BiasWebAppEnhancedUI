"# BiasWebAppEnhancedUI" 
# 🛡️ Bias Mitigation Web App

A complete pipeline for detecting and mitigating bias in machine learning models with DVC tracking, DagsHub logging, and FastAPI deployment.

## 📦 Features

- **Data Upload & Validation**: Upload datasets, define target and protected attributes
- **Bias Detection**: Comprehensive fairness metrics using AIF360
- **Fairness Mitigation**: 7 different techniques including Reweighing, Disparate Impact Remover, Adversarial Debiasing
- **Model Training**: Automated model selection with hyperparameter tuning
- **Experiment Tracking**: DVC + DagsHub + MLflow integration
- **REST API**: FastAPI endpoints for deployment

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd bias-mitigation-app

# Install dependencies
pip install -r requirements.txt
```

### 2. Initialize DVC & DagsHub

```bash
# Initialize DVC
dvc init

# Setup DagsHub (replace with your credentials)
dvc remote add origin https://dagshub.com/<username>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-username>
dvc remote modify origin --local password <your-token>

# Configure params.yaml with your DagsHub details
```

### 3. Run Pipeline

#### Option A: Using Python Scripts

```bash
# Step 1: Upload and process data
python -c "from src.data_handler import DataHandler; DataHandler().process_data()"

# Step 2: Upload model and generate biased predictions
python -c "from src.model_trainer import ModelTrainer; ..."

# Step 3: Analyze bias
python -c "from src.bias_detector import BiasDetector; BiasDetector().analyze()"

# Step 4: Apply fairness mitigation
python -c "from src.fairness_engine import FairnessEngine; FairnessEngine().apply_mitigation()"
```

#### Option B: Using DVC Pipeline

```bash
# Run entire pipeline
dvc repro

# Push results to DagsHub
dvc push
git add .
git commit -m "Run bias mitigation pipeline"
git push
```

### 4. Start API Server

```bash
# Start FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## 📡 API Endpoints

### Core Endpoints

- `POST /upload-data` - Upload dataset CSV
- `POST /set-metadata` - Define target, protected attributes, features
- `POST /upload-model` - Upload pre-trained model for bias analysis
- `GET /analyze-bias` - Run bias detection and get fairness metrics
- `POST /apply-mitigation` - Apply fairness technique
- `GET /compare-models` - Compare biased vs fair models
- `GET /model-card` - Generate model documentation

### Download Endpoints

- `GET /download-plot/{plot_name}` - Download visualization
- `GET /download-predictions/{pred_type}` - Download predictions (biased/fair)

## 🧪 Example API Usage

```python
import requests

# Upload data
with open('data.csv', 'rb') as f:
    response = requests.post('http://localhost:8000/upload-data', files={'file': f})
    print(response.json())

# Set metadata
metadata = {
    "target": "income",
    "protected": ["gender", "race"],
    "features": ["age", "education", "hours_per_week"]
}
response = requests.post('http://localhost:8000/set-metadata', json=metadata)

# Analyze bias
response = requests.get('http://localhost:8000/analyze-bias')
print(response.json()['metrics'])

# Apply mitigation
mitigation = {
    "technique": "2",  # Reweighing
    "protected_attribute": "gender",
    "encoding_method": "label"
}
response = requests.post('http://localhost:8000/apply-mitigation', json=mitigation)
```

## 🎯 Fairness Techniques

| ID | Technique | Type | Description |
|----|-----------|------|-------------|
| 1 | None | - | No mitigation (baseline) |
| 2 | Reweighing | Pre-processing | Adjust sample weights to balance outcomes |
| 3 | Disparate Impact | Pre-processing | Repair features to reduce discrimination |
| 4 | Reject Option | Post-processing | Adjust predictions near decision boundary |
| 5 | Equalized Odds | Post-processing | Balance TPR and FPR across groups |
| 6 | MetaFair | In-processing | Train with fairness constraints |
| 7 | Adversarial Debiasing | In-processing | Use adversarial network to remove bias |

## 📊 Fairness Metrics

- **Disparate Impact**: Ratio of favorable outcomes (ideal ≈ 1)
- **Statistical Parity Difference**: Difference in selection rates (ideal ≈ 0)
- **Equal Opportunity Difference**: Difference in TPR (ideal ≈ 0)
- **Average Odds Difference**: Average of TPR and FPR differences (ideal ≈ 0)
- **Theil Index**: Inequality measure (lower is better)

## 🗂️ Project Structure

```
bias-mitigation-app/
├── src/
│   ├── data_handler.py      # Data loading and preprocessing
│   ├── bias_detector.py     # Bias analysis and metrics
│   ├── fairness_engine.py   # Fairness mitigation techniques
│   ├── model_trainer.py     # Model training and selection
│   └── utils.py             # Utilities and MLflow integration
├── api.py                   # FastAPI application
├── params.yaml              # Configuration parameters
├── dvc.yaml                 # DVC pipeline definition
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

Edit `params.yaml` to configure:

- File paths for data, models, outputs
- DagsHub repository details
- Training parameters (test size, CV folds)
- Available fairness techniques

## 📈 Experiment Tracking

All experiments are tracked in DagsHub with MLflow:

- Model parameters and hyperparameters
- Fairness metrics for each protected attribute
- Training metrics (RMSE, accuracy)
- Applied mitigation techniques

View experiments at: `https://dagshub.com/<username>/<repo>`

## 🚢 Deployment

### Local Development

```bash
uvicorn api:app --reload
```

### Production (Render/Heroku)

1. Add `Procfile`:
```
web: uvicorn api:app --host 0.0.0.0 --port $PORT
```

2. Deploy to Render:
   - Connect GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

For questions or support, please open an issue on GitHub.