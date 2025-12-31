import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils import load_params, load_metadata, save_metadata, ensure_dir

class DataHandler:
    def __init__(self):
        self.params = load_params()
        
    def process_data(self):
        """Load, validate and save processed data with metadata"""
        print("📂 Loading dataset...")
        
        # Load data
        df = pd.read_csv(self.params["paths"]["raw_data"])
        print(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info
        print("\n📊 Dataset Info:")
        print(df.head())
        print("\n", df.describe())
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️ Missing values found:\n{missing[missing > 0]}")
        
        # Get column info
        print("\n🎯 Define columns:")
        print("Available columns:", list(df.columns))
        
        target = input("Enter target column: ").strip()
        protected = input("Enter protected attributes (comma-separated): ").strip().split(",")
        protected = [p.strip() for p in protected if p.strip()]
        features = input("Enter feature columns (comma-separated): ").strip().split(",")
        features = [f.strip() for f in features if f.strip() and f.strip() != target]
        
        # Validate columns
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")
        
        for p in protected:
            if p not in df.columns:
                raise ValueError(f"Protected attribute '{p}' not found")
        
        # Save processed data
        ensure_dir(self.params["paths"]["processed_data"])
        df.to_csv(self.params["paths"]["processed_data"], index=False)
        
        # Save metadata
        metadata = {
            "target": target,
            "protected": protected,
            "features": features,
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        save_metadata(metadata)
        
        print(f"\n✅ Data and metadata saved successfully")
        return df, metadata
    
    def encode_data(self, df, features, method="label"):
        """Encode features for modeling"""
        X = df[features].copy()
        
        if method == "onehot":
            X = pd.get_dummies(X, drop_first=True)
        elif method == "label":
            for col in X.select_dtypes(include=["object", "category"]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        return X
    
    def get_train_data(self):
        """Load processed data for training"""
        from src.utils import load_metadata  # Import here to ensure it's available
        
        df = pd.read_csv(self.params["paths"]["processed_data"])
        meta = load_metadata()
        
        X = self.encode_data(df, meta["features"])
        y = df[meta["target"]]
        
        return X, y, df, meta